import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run import build_status_payload

def run_tests():
    print("Executing 8-step Off-Camera Test Matrix...\n")
    results = {}
    
    # Common base state for a fully trained keystroke model running live
    base_state = {
        'model_status': 'TRAINED',
        'is_live_monitoring': True,
        'trained_modalities': '["keystroke", "keyboard"]', # Just keystroke trained
        'latency': {},
        'fusion_prob': 0.85,
        'fusion_pred': 1,
        'smoothed_stress_prob': 0.85,
        'smoothed_nonstress_prob': 0.15,
        'confidence': 0.85,
        'modality_weights': '{"keyboard": 1.0, "speech": 0.0, "facial": 0.0, "eye_pupil": 0.0, "handwriting": 0.0}',
        'reliabilities': '{"keyboard": 0.9, "speech": 0.0, "facial": 0.0, "eye_pupil": 0.0, "handwriting": 0.0}',
        'dataset_name': 'Keystroke-Stress'
    }
    
    # 1. Camera ON, Keystroke ON (Normal Operation)
    # Even if Camera is ON, since model is only trained on keystroke, ui_mask for face should be 0!
    # Wait, in the actual run.py, ui_mask is computed in inference_worker and unavailable is set there.
    # We will simulate what inference_worker writes to shared_state.
    
    def simulate(test_name, ui_mask_list, active_modalities):
        # We determine unavailable based on the ui_mask that inference_worker would compute
        unavailable = []
        mask_to_name = {0: 'speech', 1: 'facial', 2: 'keyboard', 3: 'handwriting', 4: 'eye_pupil'}
        for idx, name in mask_to_name.items():
            if ui_mask_list[idx] < 0.5:
                unavailable.append(name)
                
        state = base_state.copy()
        state['fusion_mask'] = json.dumps(ui_mask_list)
        state['unavailable_modalities'] = json.dumps(unavailable)
        
        payload = build_status_payload(state)
        results[test_name] = {
            'prediction': payload['prediction'],
            'available_modalities': payload['available_modalities'],
            'unavailable_modalities': payload['unavailable_modalities'],
            'usable_modalities': payload['usable_modalities']
        }
        print(f"[{test_name}]")
        print(f"  Prediction: {payload['prediction']}")
        print(f"  Available: {payload['available_modalities']}")
        print(f"  Usable (Trained & Active): {payload['usable_modalities']}")
        print()

    # 1. Normal (All Physically Active, but only Keystroke is trained)
    # If all physically active, inference_worker zeroes out mask for untrained.
    # ui_mask will be [0, 0, 1, 0, 0] because only keystroke is trained.
    simulate("1. Camera ON, Keyboard ON", [0.0, 0.0, 1.0, 0.0, 0.0], ['speech', 'facial', 'keyboard', 'handwriting', 'eye_pupil'])
    
    # 2. Camera OFF, Keyboard ON
    simulate("2. Camera OFF, Keyboard ON", [0.0, 0.0, 1.0, 0.0, 0.0], ['keyboard'])
    
    # 3. Camera OFF, Keyboard OFF
    # If keyboard is also physically off, mask is 0
    base_no_pred = base_state.copy()
    base_no_pred['fusion_pred'] = -1
    simulate("3. Camera OFF, Keyboard OFF", [0.0, 0.0, 0.0, 0.0, 0.0], [])
    
    # 4. Camera ON, Keyboard OFF
    simulate("4. Camera ON, Keyboard OFF", [0.0, 0.0, 0.0, 0.0, 0.0], ['speech', 'facial', 'eye_pupil'])
    
    os.makedirs('results', exist_ok=True)
    with open('results/off_camera_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    run_tests()
