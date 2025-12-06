import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Quick device check
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Devices:", tf.config.list_physical_devices())

# ---- Simple Tensor Test ----

# 4x4 constant
x = tf.constant([[1.0, 2.0, 3.0, 4.0],
                 [5.0, 6.0, 7.0, 8.0],
                 [9.0, 10.0, 11.0, 12.0],
                 [13.0, 14.0, 15.0, 16.0]])

w = tf.constant([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8],
                 [0.9, 1.0, 1.1, 1.2],
                 [1.3, 1.4, 1.5, 1.6]])

y = tf.matmul(x, w)

print("Output y:")
print(y)

# Show where TF placed ops (CPU or GPU)
print("\nDevice placement test:")
@tf.function
def matmul_test(a, b):
    return tf.matmul(a, b)

matmul_test(x, w)
print(matmul_test.experimental_get_tracing_count(), "traces executed")
