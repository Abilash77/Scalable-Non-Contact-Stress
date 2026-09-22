import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

def generate_session_pdf(session_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=10,
        alignment=1 # Center
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=20,
        alignment=1
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#003366"),
        spaceBefore=15,
        spaceAfter=10
    )
    normal_style = styles['Normal']
    
    story = []
    
    # Title
    story.append(Paragraph("NON-CONTACT STRESS DETECTION SESSION REPORT", title_style))
    story.append(Paragraph("Scalable Non-Contact Stress Detection Using Hybrid Multimodal Intelligence", subtitle_style))
    
    # Session Information
    story.append(Paragraph("SESSION INFORMATION", heading_style))
    session_info = [
        ["Session ID", session_data.get('session_id', 'N/A')],
        ["Date", session_data.get('date', 'N/A')],
        ["Start Time", session_data.get('start_time', 'N/A')],
        ["End Time", session_data.get('end_time', 'N/A')],
        ["Duration", session_data.get('duration', 'N/A')]
    ]
    t_session = Table(session_info, colWidths=[150, 350])
    t_session.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_session)
    
    # Final Stress Result
    story.append(Paragraph("FINAL STRESS RESULT", heading_style))
    stress_info = [
        ["Final Prediction (Primary)", session_data.get('final_prediction', 'INSUFFICIENT DATA')],
        ["Primary Stress Score (Camera)", f"{session_data.get('stress_probability', 0.0):.2f}%"],
        ["Primary Confidence (Camera)", f"{session_data.get('confidence', 0.0):.2f}%"],
        ["Secondary Prediction (Keyboard)", session_data.get('secondary_prediction', 'N/A')],
        ["Secondary Stress Prob (Keyboard)", f"{session_data.get('secondary_stress_prob', 0.0):.2f}%"],
        ["Secondary Confidence (Keyboard)", f"{session_data.get('secondary_confidence', 0.0):.2f}%"],
        ["Prediction Source", session_data.get('prediction_source', 'N/A')],
        ["Number of Valid Camera Predictions", str(session_data.get('valid_predictions_count', 0))],
        ["Camera Stress Predictions", str(session_data.get('stress_count', 0))],
        ["Camera Non-Stress Predictions", str(session_data.get('nonstress_count', 0))],
        ["Average Camera Stress Score", f"{session_data.get('avg_stress_prob', 0.0):.2f}%"],
        ["Peak Camera Stress Score", f"{session_data.get('peak_stress_prob', 0.0):.2f}%"],
        ["Average Camera Confidence", f"{session_data.get('avg_confidence', 0.0):.2f}%"]
    ]
    t_stress = Table(stress_info, colWidths=[150, 350])
    t_stress.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_stress)
    
    # Five Modality Results
    story.append(Paragraph("FIVE MODALITY RESULTS", heading_style))
    mods = session_data.get('modalities', {})
    
    # Keyboard
    story.append(Paragraph("1. KEYBOARD", styles['Heading3']))
    kb = mods.get('keyboard', {})
    kb_data = [
        ["Status", kb.get('status', 'OFFLINE')],
        ["7D feature vector", "YES" if kb.get('features_ready') else "NO"],
        ["Valid windows", str(kb.get('buffer_fill', 0))],
        ["Prediction contribution", "STRESS MODEL INPUT"]
    ]
    t_kb = Table(kb_data, colWidths=[150, 350])
    t_kb.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f9f9f9"))]))
    story.append(t_kb)
    story.append(Spacer(1, 10))
    
    # Speech
    story.append(Paragraph("2. SPEECH", styles['Heading3']))
    sp = mods.get('speech', {})
    sp_data = [
        ["Status", sp.get('status', 'OFFLINE')],
        ["169D feature vector", "YES" if sp.get('features_ready') else "NO"],
        ["Valid windows", str(sp.get('buffer_fill', 0))],
        ["Prediction contribution", "FEATURE EXTRACTION ONLY"]
    ]
    t_sp = Table(sp_data, colWidths=[150, 350])
    t_sp.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f9f9f9"))]))
    story.append(t_sp)
    story.append(Spacer(1, 10))
    
    # Face
    story.append(Paragraph("3. FACIAL EXPRESSION", styles['Heading3']))
    fc = mods.get('facial', {})
    fc_data = [
        ["Status", fc.get('status', 'OFFLINE')],
        ["Face detected count", str(fc.get('face_detection_count', 0))],
        ["Latest expression", fc.get('latest_expression', 'N/A')],
        ["Expression confidence", str(fc.get('expression_confidence', 'N/A'))],
        ["Prediction contribution", "FEATURE EXTRACTION ONLY"]
    ]
    t_fc = Table(fc_data, colWidths=[150, 350])
    t_fc.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f9f9f9"))]))
    story.append(t_fc)
    story.append(Spacer(1, 10))
    
    # Eye
    story.append(Paragraph("4. EYE / PUPIL", styles['Heading3']))
    ey = mods.get('eye_pupil', {})
    ey_data = [
        ["Status", ey.get('status', 'OFFLINE')],
        ["5D feature vector", "YES" if ey.get('features_ready') else "NO"],
        ["Valid windows", str(ey.get('buffer_fill', 0))],
        ["Prediction contribution", "FEATURE EXTRACTION ONLY"]
    ]
    t_ey = Table(ey_data, colWidths=[150, 350])
    t_ey.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f9f9f9"))]))
    story.append(t_ey)
    story.append(Spacer(1, 10))
    
    # Handwriting
    story.append(Paragraph("5. HANDWRITING", styles['Heading3']))
    hw = mods.get('handwriting', {})
    hw_data = [
        ["Status", hw.get('status', 'OFFLINE')],
        ["9D feature vector", "YES" if hw.get('features_ready') else "NO"],
        ["Valid submissions/windows", str(hw.get('buffer_fill', 0))],
        ["Prediction contribution", "FEATURE EXTRACTION ONLY"]
    ]
    t_hw = Table(hw_data, colWidths=[150, 350])
    t_hw.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f9f9f9"))]))
    story.append(t_hw)
    
    # Prediction Timeline
    story.append(Paragraph("PREDICTION TIMELINE", heading_style))
    timeline_data = session_data.get('timeline', [])
    if len(timeline_data) >= 2:
        plt.figure(figsize=(6, 4))
        times = [datetime.datetime.strptime(x['timestamp'], "%H:%M:%S") for x in timeline_data if 'camera_stress_prob_pct' in x]
        
        if times:
            # Primary Camera
            cam_stress = [x['camera_stress_prob_pct'] for x in timeline_data if 'camera_stress_prob_pct' in x]
            plt.plot(times, cam_stress, label='Primary (Camera) %', color='red', marker='o')
            
            # Secondary Keyboard
            key_times = [datetime.datetime.strptime(x['timestamp'], "%H:%M:%S") for x in timeline_data if 'keyboard_stress_prob_pct' in x]
            key_stress = [x['keyboard_stress_prob_pct'] for x in timeline_data if 'keyboard_stress_prob_pct' in x]
            if key_times:
                plt.plot(key_times, key_stress, label='Secondary (Keyboard) %', color='blue', marker='x', linestyle='--')
            
            plt.title('Real-time Prediction Timeline')
            plt.xlabel('Time')
            plt.ylabel('Stress Probability (%)')
            plt.legend()
            plt.grid(True)
            import matplotlib.dates as mdates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gcf().autofmt_xdate()
            
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight')
            plt.close()
            img_buf.seek(0)
            story.append(Image(img_buf, width=400, height=266))
        else:
            story.append(Paragraph("Insufficient valid prediction history for timeline.", normal_style))
    else:
        story.append(Paragraph("Insufficient valid prediction history for timeline.", normal_style))
    
    # Modality Activity Summary
    story.append(Paragraph("MODALITY ACTIVITY SUMMARY", heading_style))
    mod_summary = [["Modality", "Status", "Valid Windows", "Feature Size"]]
    for m in ['keyboard', 'speech', 'facial', 'eye_pupil', 'handwriting']:
        d = mods.get(m, {})
        mod_summary.append([
            m.capitalize(),
            d.get('status', 'OFFLINE'),
            str(d.get('buffer_fill', 0)),
            d.get('feature_shape', 'N/A')
        ])
    t_mod_sum = Table(mod_summary, colWidths=[100, 100, 100, 100])
    t_mod_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#003366")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t_mod_sum)
    
    # Data Quality
    story.append(Paragraph("DATA QUALITY", heading_style))
    dq = session_data.get('data_quality', {})
    dq_info = [
        ["Total session duration", session_data.get('duration', 'N/A')],
        ["Valid prediction windows", str(session_data.get('valid_predictions_count', 0))],
        ["Invalid/waiting windows", str(dq.get('failed_cycles', 0))],
        ["Camera availability", "YES" if dq.get('camera_connected') else "NO"],
        ["Keyboard availability", "YES" if mods.get('keyboard', {}).get('status') != 'OFFLINE' else "NO"],
        ["Handwriting submissions", str(mods.get('handwriting', {}).get('buffer_fill', 0))]
    ]
    t_dq = Table(dq_info, colWidths=[200, 300])
    t_dq.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f2f2f2"))]))
    story.append(t_dq)
    
    # Model Information
    story.append(Paragraph("MODEL INFORMATION", heading_style))
    model_info = [
        ["Model", session_data.get('model_name', 'N/A')],
        ["Prediction source", session_data.get('prediction_source', 'N/A')],
        ["Input feature size", "7D (Keyboard)"],
        ["Window", "10 samples"],
        ["Classification", "STRESS / NON-STRESS"]
    ]
    t_model = Table(model_info, colWidths=[150, 350])
    t_model.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f2f2f2"))]))
    story.append(t_model)
    
    # Research Disclaimer
    story.append(Spacer(1, 30))
    story.append(Paragraph("<i>This system is an experimental research prototype for stress-related signal analysis and is not a medical diagnostic system.</i>", normal_style))
    
    doc.build(story)
    return output_path
