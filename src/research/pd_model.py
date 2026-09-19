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

def focal_calibration_loss(y_true, y_pred, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    y_true_int = tf.cast(tf.squeeze(y_true), tf.int32)
    y_true_one_hot = tf.one_hot(y_true_int, depth=tf.shape(y_pred)[-1])
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    focal_loss = -tf.math.pow((1.0 - p_t), gamma) * tf.math.log(p_t)
    return tf.reduce_mean(focal_loss)

def custom_pd_loss(lambda2=0.01):
    def pd_output_loss(y_true, y_pred):
        l_ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        l_cal = focal_calibration_loss(y_true, y_pred)
        return l_ce + lambda2 * l_cal
    return pd_output_loss

def build_pd_model(input_shape=(10, 5), gru_units=64, num_classes=2):
    """
    Unimodal Pupil Diameter (PD) Model matching the architecture of exp201-exp205 
    for ForDigitStress from the project documentation.
    """
    inputs = Input(shape=input_shape, name="input_eye")
    
    branch = ModalityBranch(encoder_units=128, gru_units=gru_units, name="branch_eye")
    hm = branch(inputs)
    
    outputs = Dense(num_classes, activation='softmax', name="pd_output", kernel_regularizer=L2(1e-4))(hm)
    
    model = Model(inputs=inputs, outputs=outputs, name="PD_Unimodal_Model")
    return model
