import numpy as np
from numpy import linalg as la
import tensorly as tl
import numpy as np
import tensorflow as tf



unfold_a = tl.unfold(defusion_matrix, mode=0)
print(unfold_a.shape)
unfold_b = tl.unfold(defusion_matrix, mode=1)
print(unfold_b.shape)
unfold_c = tl.unfold(defusion_matrix, mode=2)
print(unfold_c.shape)
unfold_d = tl.unfold(defusion_matrix, mode=3)
print(unfold_d.shape)

print(unfold_a.dtype, unfold_b.dtype, unfold_c.dtype, unfold_d.dtype)

# SVD
S_a, U_a, V_a = tf.linalg.svd(unfold_a)
print(U_a.dtype)
print(U_a)
print(U_a.shape)

S_b, U_b, V_b = tf.linalg.svd(unfold_b)
print(U_b.shape)
S_c, U_c, V_c = tf.linalg.svd(unfold_c)
print(U_c.shape)
S_d, U_d, V_d = tf.linalg.svd(unfold_d)
print(U_d.shape)