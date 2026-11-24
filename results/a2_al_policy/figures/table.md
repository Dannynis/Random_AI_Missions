| features | task | learner | method | AUC (mean ± std over seeds) | seeds | policy vs best baseline |
|---|---|---|---|---|---|---|
| dinov2-base | cifar10 | linear | tcm100_badge | 0.9202 ± 0.0077 | 10 | |
| dinov2-base | cifar10 | linear | policy2_bc | 0.9198 ± 0.0076 | 10 | vs tcm100_badge: -0.0005, wins 2/10 |
| dinov2-base | cifar10 | linear | tcm100 | 0.9197 ± 0.0076 | 10 | |
| dinov2-base | cifar10 | linear | typiclust | 0.9197 ± 0.0076 | 10 | |
| dinov2-base | cifar10 | linear | tcm300 | 0.9194 ± 0.0076 | 10 | |
| dinov2-base | cifar10 | linear | policy2 | 0.9128 ± 0.0061 | 10 | vs tcm100_badge: -0.0074, wins 1/10 |
| dinov2-base | cifar10 | linear | margin | 0.8889 ± 0.0208 | 10 | |
| dinov2-base | cifar10 | linear | badge | 0.8884 ± 0.0206 | 10 | |
| dinov2-base | cifar10 | linear | random | 0.8787 ± 0.0139 | 10 | |
| dinov2-base | cifar10 | linear | policy | 0.8581 ± 0.0157 | 10 | vs tcm100_badge: -0.0621, wins 0/10 |
| dinov2-base | cifar10 | linear | coreset | 0.8498 ± 0.0167 | 10 | |
| dinov2-base | cifar10 | linear | entropy | 0.8227 ± 0.0419 | 10 | |
| dinov2-base | cifar10 | linear | probcover | 0.8207 ± 0.0241 | 10 | |
| dinov2-base | cifar100 | linear | typiclust | 0.5362 ± 0.0051 | 10 | |
| dinov2-base | cifar100 | linear | policy2_bc | 0.5359 ± 0.0050 | 10 | vs typiclust: -0.0003, wins 2/10 |
| dinov2-base | cifar100 | linear | tcm300 | 0.5340 ± 0.0051 | 10 | |
| dinov2-base | cifar100 | linear | policy2 | 0.5290 ± 0.0054 | 10 | vs typiclust: -0.0071, wins 0/10 |
| dinov2-base | cifar100 | linear | tcm100_badge | 0.5237 ± 0.0053 | 10 | |
| dinov2-base | cifar100 | linear | tcm100 | 0.5229 ± 0.0036 | 10 | |
| dinov2-base | cifar100 | linear | probcover | 0.4747 ± 0.0031 | 10 | |
| dinov2-base | cifar100 | linear | policy | 0.4677 ± 0.0079 | 10 | vs typiclust: -0.0685, wins 0/10 |
| dinov2-base | cifar100 | linear | coreset | 0.4528 ± 0.0047 | 10 | |
| dinov2-base | cifar100 | linear | margin | 0.4488 ± 0.0079 | 10 | |
| dinov2-base | cifar100 | linear | badge | 0.4458 ± 0.0114 | 10 | |
| dinov2-base | cifar100 | linear | random | 0.4209 ± 0.0101 | 10 | |
| dinov2-base | cifar100 | linear | entropy | 0.3672 ± 0.0111 | 10 | |
| dinov2-base | c100_heldout | linear | tcm100_badge | 0.9439 ± 0.0026 | 10 | |
| dinov2-base | c100_heldout | linear | tcm100 | 0.9435 ± 0.0027 | 10 | |
| dinov2-base | c100_heldout | linear | tcm300 | 0.9432 ± 0.0028 | 10 | |
| dinov2-base | c100_heldout | linear | policy2_bc | 0.9428 ± 0.0029 | 10 | vs tcm100_badge: -0.0011, wins 1/10 |
| dinov2-base | c100_heldout | linear | typiclust | 0.9426 ± 0.0029 | 10 | |
| dinov2-base | c100_heldout | linear | policy2 | 0.9394 ± 0.0031 | 10 | vs tcm100_badge: -0.0045, wins 0/10 |
| dinov2-base | c100_heldout | linear | probcover | 0.9257 ± 0.0050 | 10 | |
| dinov2-base | c100_heldout | linear | badge | 0.9006 ± 0.0043 | 10 | |
| dinov2-base | c100_heldout | linear | policy | 0.8972 ± 0.0044 | 10 | vs tcm100_badge: -0.0467, wins 0/10 |
| dinov2-base | c100_heldout | linear | coreset | 0.8969 ± 0.0055 | 10 | |
| dinov2-base | c100_heldout | linear | margin | 0.8946 ± 0.0059 | 10 | |
| dinov2-base | c100_heldout | linear | random | 0.8826 ± 0.0066 | 10 | |
| dinov2-base | c100_heldout | linear | entropy | 0.8548 ± 0.0131 | 10 | |
| dinov2-base | cifar10 | lp | policy2_bc | 0.9528 ± 0.0053 | 10 | vs typiclust: +0.0000, wins 3/10 |
| dinov2-base | cifar10 | lp | typiclust | 0.9528 ± 0.0053 | 10 | |
| dinov2-base | cifar10 | lp | tcm300 | 0.9516 ± 0.0052 | 10 | |
| dinov2-base | cifar10 | lp | tcm100_badge | 0.9493 ± 0.0052 | 10 | |
| dinov2-base | cifar10 | lp | tcm100 | 0.9472 ± 0.0053 | 10 | |
| dinov2-base | cifar10 | lp | policy2 | 0.9367 ± 0.0062 | 10 | vs typiclust: -0.0161, wins 0/10 |
| dinov2-base | cifar10 | lp | random | 0.9116 ± 0.0136 | 10 | |
| dinov2-base | cifar10 | lp | badge | 0.9068 ± 0.0176 | 10 | |
| dinov2-base | cifar10 | lp | margin | 0.9013 ± 0.0172 | 10 | |
| dinov2-base | cifar10 | lp | coreset | 0.9005 ± 0.0090 | 10 | |
| dinov2-base | cifar10 | lp | policy | 0.8880 ± 0.0122 | 10 | vs typiclust: -0.0648, wins 0/10 |
| dinov2-base | cifar10 | lp | probcover | 0.8759 ± 0.0160 | 10 | |
| dinov2-base | cifar10 | lp | entropy | 0.8589 ± 0.0400 | 10 | |
| dinov2-base | cifar100 | lp | typiclust | 0.5610 ± 0.0038 | 10 | |
| dinov2-base | cifar100 | lp | policy2_bc | 0.5610 ± 0.0037 | 10 | vs typiclust: -0.0001, wins 4/10 |
| dinov2-base | cifar100 | lp | tcm300 | 0.5587 ± 0.0040 | 10 | |
| dinov2-base | cifar100 | lp | policy2 | 0.5577 ± 0.0045 | 10 | vs typiclust: -0.0033, wins 0/10 |
| dinov2-base | cifar100 | lp | tcm100_badge | 0.5475 ± 0.0042 | 10 | |
| dinov2-base | cifar100 | lp | tcm100 | 0.5463 ± 0.0035 | 10 | |
| dinov2-base | cifar100 | lp | coreset | 0.5024 ± 0.0041 | 10 | |
| dinov2-base | cifar100 | lp | probcover | 0.4980 ± 0.0028 | 10 | |
| dinov2-base | cifar100 | lp | policy | 0.4939 ± 0.0075 | 10 | vs typiclust: -0.0671, wins 0/10 |
| dinov2-base | cifar100 | lp | margin | 0.4816 ± 0.0093 | 10 | |
| dinov2-base | cifar100 | lp | badge | 0.4814 ± 0.0115 | 10 | |
| dinov2-base | cifar100 | lp | random | 0.4603 ± 0.0084 | 10 | |
| dinov2-base | cifar100 | lp | entropy | 0.4197 ± 0.0114 | 10 | |
| dinov2-base | c100_heldout | lp | tcm300 | 0.9501 ± 0.0024 | 10 | |
| dinov2-base | c100_heldout | lp | policy2_bc | 0.9501 ± 0.0024 | 10 | vs tcm300: -0.0000, wins 6/10 |
| dinov2-base | c100_heldout | lp | typiclust | 0.9499 ± 0.0024 | 10 | |
| dinov2-base | c100_heldout | lp | tcm100_badge | 0.9487 ± 0.0026 | 10 | |
| dinov2-base | c100_heldout | lp | tcm100 | 0.9479 ± 0.0024 | 10 | |
| dinov2-base | c100_heldout | lp | policy2 | 0.9444 ± 0.0029 | 10 | vs tcm300: -0.0057, wins 0/10 |
| dinov2-base | c100_heldout | lp | probcover | 0.9354 ± 0.0040 | 10 | |
| dinov2-base | c100_heldout | lp | coreset | 0.9232 ± 0.0062 | 10 | |
| dinov2-base | c100_heldout | lp | badge | 0.9109 ± 0.0051 | 10 | |
| dinov2-base | c100_heldout | lp | margin | 0.9051 ± 0.0058 | 10 | |
| dinov2-base | c100_heldout | lp | random | 0.9044 ± 0.0068 | 10 | |
| dinov2-base | c100_heldout | lp | policy | 0.9041 ± 0.0057 | 10 | vs tcm300: -0.0460, wins 0/10 |
| dinov2-base | c100_heldout | lp | entropy | 0.8778 ± 0.0105 | 10 | |
| dinov2-small | cifar10 | linear | typiclust | 0.8974 ± 0.0115 | 10 | |
| dinov2-small | cifar10 | linear | badge | 0.8596 ± 0.0173 | 10 | |
| dinov2-small | cifar10 | linear | margin | 0.8578 ± 0.0200 | 10 | |
| dinov2-small | cifar10 | linear | random | 0.8461 ± 0.0153 | 10 | |
| dinov2-small | cifar10 | linear | coreset | 0.8254 ± 0.0117 | 10 | |
| dinov2-small | cifar10 | linear | entropy | 0.8154 ± 0.0147 | 10 | |
| dinov2-small | cifar10 | linear | probcover | 0.7810 ± 0.0243 | 10 | |
| dinov2-small | cifar100 | linear | typiclust | 0.4738 ± 0.0049 | 10 | |
| dinov2-small | cifar100 | linear | probcover | 0.3937 ± 0.0206 | 10 | |
| dinov2-small | cifar100 | linear | coreset | 0.3888 ± 0.0056 | 10 | |
| dinov2-small | cifar100 | linear | margin | 0.3866 ± 0.0066 | 10 | |
| dinov2-small | cifar100 | linear | badge | 0.3835 ± 0.0107 | 10 | |
| dinov2-small | cifar100 | linear | random | 0.3683 ± 0.0090 | 10 | |
| dinov2-small | cifar100 | linear | entropy | 0.3374 ± 0.0150 | 10 | |
| dinov2-small | c100_heldout | linear | typiclust | 0.9162 ± 0.0026 | 10 | |
| dinov2-small | c100_heldout | linear | probcover | 0.8851 ± 0.0071 | 10 | |
| dinov2-small | c100_heldout | linear | coreset | 0.8749 ± 0.0064 | 10 | |
| dinov2-small | c100_heldout | linear | margin | 0.8737 ± 0.0072 | 10 | |
| dinov2-small | c100_heldout | linear | badge | 0.8734 ± 0.0072 | 10 | |
| dinov2-small | c100_heldout | linear | random | 0.8543 ± 0.0063 | 10 | |
| dinov2-small | c100_heldout | linear | entropy | 0.8336 ± 0.0122 | 10 | |
| dinov2-small | cifar10 | lp | typiclust | 0.9245 ± 0.0055 | 10 | |
| dinov2-small | cifar10 | lp | random | 0.8787 ± 0.0135 | 10 | |
| dinov2-small | cifar10 | lp | badge | 0.8680 ± 0.0200 | 10 | |
| dinov2-small | cifar10 | lp | margin | 0.8663 ± 0.0218 | 10 | |
| dinov2-small | cifar10 | lp | coreset | 0.8614 ± 0.0103 | 10 | |
| dinov2-small | cifar10 | lp | entropy | 0.8308 ± 0.0230 | 10 | |
| dinov2-small | cifar10 | lp | probcover | 0.8174 ± 0.0236 | 10 | |
| dinov2-small | cifar100 | lp | typiclust | 0.5016 ± 0.0052 | 10 | |
| dinov2-small | cifar100 | lp | coreset | 0.4429 ± 0.0057 | 10 | |
| dinov2-small | cifar100 | lp | margin | 0.4260 ± 0.0070 | 10 | |
| dinov2-small | cifar100 | lp | badge | 0.4239 ± 0.0115 | 10 | |
| dinov2-small | cifar100 | lp | probcover | 0.4204 ± 0.0189 | 10 | |
| dinov2-small | cifar100 | lp | random | 0.4089 ± 0.0092 | 10 | |
| dinov2-small | cifar100 | lp | entropy | 0.3855 ± 0.0125 | 10 | |
| dinov2-small | c100_heldout | lp | typiclust | 0.9259 ± 0.0016 | 10 | |
| dinov2-small | c100_heldout | lp | coreset | 0.9025 ± 0.0046 | 10 | |
| dinov2-small | c100_heldout | lp | probcover | 0.8949 ± 0.0074 | 10 | |
| dinov2-small | c100_heldout | lp | badge | 0.8866 ± 0.0060 | 10 | |
| dinov2-small | c100_heldout | lp | margin | 0.8815 ± 0.0056 | 10 | |
| dinov2-small | c100_heldout | lp | random | 0.8790 ± 0.0065 | 10 | |
| dinov2-small | c100_heldout | lp | entropy | 0.8566 ± 0.0088 | 10 | |
