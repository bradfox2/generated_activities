dt_src.shape
embedded_src.shape
src.shape


em_st_src.shape
se_data_ten.shape
se(st_data_ten).shape

A = se(st_data_ten).flatten(1)

A.repeat(4, 1).shape

A = A.unsqueeze(0).repeat(4, 1, 1)

A.shape

tfmr_enc_out.shape
mask.shape

em_st_src.shape
em_st_src[1]

dt_src.shape
tfmr_enc_out.shape

tfmr_out.shape


dt_src.shape
tfmr_enc_out.shape
mask.shape

src.shape
src[0]
src[1]
