import numpy as np
def optimization(df_m, df_c):
  mt_c = np.matrix(df_c)
  mt_m = np.matrix(df_m)
  mt_delta = np.linalg.inv(mt_m.T @ mt_m) @ (mt_m.T @ mt_c)
  return mt_delta