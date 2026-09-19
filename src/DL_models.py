import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Layer

from tensorflow.keras.regularizers import L2

class ModalityBranch(Layer):
    def __init__(self, encoder_units, gru_units, name=None):
        super(ModalityBranch, self).__init__(name=name)
        self.encoder_dense1 = Dense(encoder_units, activation='relu', kernel_regularizer=L2(1e-4))
        self.encoder_drop = Dropout(0.3)
        self.temporal_gru = GRU(gru_units, return_sequences=False, kernel_regularizer=L2(1e-4))
        
    def call(self, inputs):
        x = self.encoder_dense1(inputs)
        x = self.encoder_drop(x)
        hm = self.temporal_gru(x)
        return hm

class ReliabilityAttention(Layer):
    def __init__(self, num_modalities, name="reliability_attention"):
        super(ReliabilityAttention, self).__init__(name=name)
        self.num_modalities = num_modalities
        self.Wr = [Dense(1, activation='sigmoid', name=f"r_{i}", kernel_regularizer=L2(1e-4)) for i in range(num_modalities)]
        self.Wa = [Dense(1, activation='linear', name=f"a_{i}", kernel_regularizer=L2(1e-4)) for i in range(num_modalities)]
        
    def call(self, hm_list, mask):
        r_list = []
        alpha_logits = []
        
        for i in range(self.num_modalities):
            hm = hm_list[i]
            rm = self.Wr[i](hm)
            r_list.append(rm)
            
            f_hm_rm = hm * rm 
            logit = self.Wa[i](f_hm_rm)
            alpha_logits.append(logit)
            
        logits_concat = tf.concat(alpha_logits, axis=-1)
        
        # Apply mask
        mask_penalty = (1.0 - mask) * -1e9
        masked_logits = logits_concat + mask_penalty
        
        alphas = tf.nn.softmax(masked_logits, axis=-1)
        return r_list, alphas

class FeatureFusion(Layer):
    def __init__(self, name="feature_fusion"):
        super(FeatureFusion, self).__init__(name=name)
        
    def call(self, inputs):
        # inputs is a tuple/list: (hm_list_stacked, alphas)
        hm_stack, alphas = inputs
        alphas_expanded = tf.expand_dims(alphas, axis=-1)
        F = tf.reduce_sum(hm_stack * alphas_expanded, axis=1)
        return F

def build_fusion_model(input_shapes, gru_units=64, num_classes=2):
    modality_names = ['audio', 'face', 'keystroke', 'handwriting', 'eye']
    num_modalities = len(modality_names)
    
    inputs = {}
    for m in modality_names:
        inputs[f"input_{m}"] = Input(shape=input_shapes[m], name=f"input_{m}")
        
    mask_input = Input(shape=(num_modalities,), name="input_mask", dtype=tf.float32)
    inputs['input_mask'] = mask_input
    
    hm_list = []
    unimodal_preds = []
    
    for i, m in enumerate(modality_names):
        branch = ModalityBranch(encoder_units=128, gru_units=gru_units, name=f"branch_{m}")
        hm = branch(inputs[f"input_{m}"])
        hm_list.append(hm)
        
        u_pred = Dense(num_classes, activation='softmax', name=f"unimodal_{m}", kernel_regularizer=L2(1e-4))(hm)
        unimodal_preds.append(u_pred)
        
    rel_att = ReliabilityAttention(num_modalities)
    r_list, alphas = rel_att(hm_list, mask_input)
    
    # Use custom layer for stack and reduce_sum to avoid KerasTensor raw TF op errors
    hm_stack = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(hm_list)
    F = FeatureFusion()([hm_stack, alphas])
    
    # Eq 7: y_hat = softmax(W_c * F + b_c)
    outputs = Dense(num_classes, activation='softmax', name="fusion_output", kernel_regularizer=L2(1e-4))(F)
    
    r_concat = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1), name="reliabilities")(r_list)
    
    model = Model(inputs=inputs, outputs={
        'fusion_output': outputs,
        'unimodal_audio': unimodal_preds[0],
        'unimodal_face': unimodal_preds[1],
        'unimodal_keystroke': unimodal_preds[2],
        'unimodal_handwriting': unimodal_preds[3],
        'unimodal_eye': unimodal_preds[4],
        'reliabilities': r_concat,
        'alphas': alphas
    }, name="RA_HMSD_Fusion_Model")
    
    return model

def build_swell_fusion_model(input_shapes, num_classes=2):
    # This is a specialized adapter model that takes all 5 modalities PLUS a generic SWELL branch.
    modality_names = ['audio', 'face', 'keystroke', 'handwriting', 'eye', 'swell']
    num_modalities = len(modality_names)
    
    inputs = {}
    for m in modality_names:
        if m in input_shapes:
            inputs[f"input_{m}"] = Input(shape=input_shapes[m], name=f"input_{m}")
        
    mask_input = Input(shape=(num_modalities,), name="input_mask", dtype=tf.float32)
    inputs['input_mask'] = mask_input
    
    hm_list = []
    unimodal_preds = []
    
    for i, m in enumerate(modality_names):
        # We reuse the ModalityBranch for standard modalities, and a similar one for SWELL
        branch = ModalityBranch(encoder_units=128, gru_units=64, name=f"branch_{m}")
        if f"input_{m}" in inputs:
            hm = branch(inputs[f"input_{m}"])
        else:
            # If the modality shape wasn't provided, this branch can't be used (but shouldn't happen)
            hm = tf.zeros_like(branch(Input(shape=(10,1)))) 
            
        hm_list.append(hm)
        
        u_pred = Dense(num_classes, activation='softmax', name=f"unimodal_{m}", kernel_regularizer=L2(1e-4))(hm)
        unimodal_preds.append(u_pred)
        
    rel_att = ReliabilityAttention(num_modalities)
    r_list, alphas = rel_att(hm_list, mask_input)
    
    hm_stack = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(hm_list)
    F = FeatureFusion()([hm_stack, alphas])
    
    outputs = Dense(num_classes, activation='softmax', name="fusion_output", kernel_regularizer=L2(1e-4))(F)
    r_concat = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1), name="reliabilities")(r_list)
    
    model = Model(inputs=inputs, outputs={
        'fusion_output': outputs,
        'unimodal_audio': unimodal_preds[0],
        'unimodal_face': unimodal_preds[1],
        'unimodal_keystroke': unimodal_preds[2],
        'unimodal_handwriting': unimodal_preds[3],
        'unimodal_eye': unimodal_preds[4],
        'unimodal_swell': unimodal_preds[5],
        'reliabilities': r_concat,
        'alphas': alphas
    }, name="SWELL_Adapter_Fusion_Model")
    
    return model

def focal_calibration_loss(y_true, y_pred, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    # Ensure y_true is cast appropriately for one_hot
    y_true_int = tf.cast(tf.squeeze(y_true), tf.int32)
    y_true_one_hot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[-1])
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    focal_loss = -tf.math.pow((1.0 - p_t), gamma) * tf.math.log(p_t)
    return tf.reduce_mean(focal_loss)

def custom_fusion_loss(lambda1=0.1, lambda2=0.01):
    def fusion_output_loss(y_true, y_pred):
        l_ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        l_cal = focal_calibration_loss(y_true, y_pred)
        # Note: L2 is automatically added by Keras to the total model loss during training 
        # because we added kernel_regularizer=L2() to the Dense/GRU layers.
        return l_ce + lambda2 * l_cal
        
    def unimodal_loss(y_true, y_pred):
        return lambda1 * tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
    return {
        'fusion_output': fusion_output_loss,
        'unimodal_audio': unimodal_loss,
        'unimodal_face': unimodal_loss,
        'unimodal_keystroke': unimodal_loss,
        'unimodal_handwriting': unimodal_loss,
        'unimodal_eye': unimodal_loss,
        'unimodal_swell': unimodal_loss,
    }

def get_model(model_name='fusion', input_shapes=None, num_classes=2, params=None):
    if model_name == 'fusion':
        if input_shapes is None:
            input_shapes = {
                'audio': (10, 169),
                'face': (10, 12),
                'keystroke': (10, 7),
                'handwriting': (10, 9),
                'eye': (10, 5)
            }
        return build_fusion_model(input_shapes, gru_units=64, num_classes=num_classes)
    elif model_name == 'swell_fusion':
        if input_shapes is None:
            input_shapes = {
                'audio': (10, 169),
                'face': (10, 12),
                'keystroke': (10, 7),
                'handwriting': (10, 9),
                'eye': (10, 5)
            }
        return build_swell_fusion_model(input_shapes, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
