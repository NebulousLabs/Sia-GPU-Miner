# Sia-CUDA-Miner
A GPU miner designed for mining siacoins. This miner runs in a command prompt and prints your hashrate along side the number of blocks you've mined.

## How to Use
1) Make sure you have a recent version of Sia installed and running.

2) Start the miner. It will mine blocks until killed with Ctrl-C.

You can tweak the miner settings with three command-line arguments: `-s`, `-c`, and `-p`.
 `-s` controls the time between refreshing the hash rate, and `-c` controls the number of iterations performed between each refresh.
For example, `./gpu-miner -s 0.2 -c 100` will perform 100 iterations every 0.2 seconds, while `./gpu-miner` will perform 1 iteration every 10 seconds.
**Note that "1 iteration" is not a constant amount of work. If -c is 10, each iteration will perform 10x less work than with -c 1.**
So while these parameters can have a minor effect on your hash rate, their primary function is to reduce the strain on your GPU. This can prevent your GPU from crashing and prevent your computer from freezing during mining.

Finally, if you are running siad on a non-default API port, you can use `-p` to specify the port to communicate on.

## Notes
*    Each Sia block takes about 10 minutes to mine.
*    Once a block is mined, Sia waits for 144 confirmation blocks before the reward is added to your wallet, which takes about 24 hours.
*    Sia currently doesn't have any mining pools. A p2pool portal is under development, and can be expected to be ready by the end of the summer.
*    Sia discussion: https://bitcointalk.org/index.php?topic=1060294