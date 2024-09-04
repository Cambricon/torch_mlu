import sys
import pandas as pd

log1_location = sys.argv[1]
log2_location = sys.argv[2]
epoch = sys.argv[3]
bench_size = sys.argv[4]

print(log1_location)
print(log2_location)
print(epoch)

df_cpu = pd.read_csv(
    log1_location + "/epoch_" + str(epoch) + "_" + str(bench_size) + "_1.csv",
    header=None,
)
df_mlu = pd.read_csv(
    log2_location + "/epoch_" + str(epoch) + "_" + str(bench_size) + "_1.csv",
    header=None,
)


a = df_cpu.loc[:, 1]
m = df_mlu.loc[:, 1]
corr = df_cpu.corrwith(m, axis=0)
corr_r2 = corr[1] ** 2
print("epoch:", epoch)
print("r2", corr_r2)

sum_cpu = 0.0
sum_mlu = 0.0
diff_sum = 0.0
for i in range(int(bench_size)):
    cpu_loss = df_cpu[1][i]  # 0: num, 1: loss, 2:accuracy
    mlu_loss = df_mlu[1][i]
    sum_cpu = sum_cpu + abs(cpu_loss)
    sum_mlu = sum_mlu + abs(mlu_loss)
    diff_sum = diff_sum + abs(cpu_loss - mlu_loss)

if sum_cpu != 0:
    MSE = diff_sum / sum_cpu
    print("MSE:", MSE)
else:
    print("Sum of CPU Loss == 0, cannot calculate MSE")
