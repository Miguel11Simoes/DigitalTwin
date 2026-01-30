"""
Focal Loss Implementation for Keras
Para lidar com classes desbalanceadas em severity e mode
"""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="Custom")
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss para classificação multi-classe.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Peso para cada classe (balancing factor). Default: 0.25
        gamma: Focusing parameter. Default: 2.0
               gamma=0 → standard CE
               gamma>0 → reduz peso de exemplos fáceis
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth (integer class indices)
            y_pred: Predições (probabilidades)
        """
        # Garantir que y_true é 1D
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        
        # Converter logits para probabilidades se necessário
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip para estabilidade numérica
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Criar one-hot
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, num_classes, dtype=y_pred.dtype)
        
        # Cross entropy: -log(p_t) onde p_t é a prob da classe correta
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
        
        # p_t: probabilidade da classe correta
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Focal loss: -alpha * (1-p_t)^gamma * log(p_t)
        loss = self.alpha * focal_weight * ce
        
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "from_logits": self.from_logits,
        })
        return config


def focal_loss(alpha=0.25, gamma=2.0, from_logits=False):
    """
    Factory function para criar Focal Loss.
    
    Usage:
        model.compile(
            loss={
                "severity": focal_loss(alpha=0.25, gamma=2.0),
                "mode": focal_loss(alpha=0.25, gamma=2.0),
            }
        )
    """
    return FocalLoss(alpha=alpha, gamma=gamma, from_logits=from_logits)
