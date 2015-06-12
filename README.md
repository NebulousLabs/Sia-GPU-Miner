# Sia-GPU-Miner
A GPU miner designed for mining siacoins. This miner runs in a command prompt and prints your hashrate along side the number of blocks you've mined.

## How to Use
1) Build the miner by running `make`.

2) Make sure you have a recent version of Sia installed and running.

3) Run the miner by running `./gpu-miner`. It will mine blocks until killed with Ctrl-C.

##Configuration
You can tweak the miner settings with five command-line arguments: `-I`, `-C`, `-p`, `-d`, and `-P`.
* -I controls the intensity. On each GPU call, the GPU will do 2^I hashes. The default is 16.
* -C controls how many iterations are done between updating the hash on your screen. Increase this if the hash is printing too fast to read.
* -p allows you to pick a platform. Default is the first platform (indexing from 0).
* -d allows you to pick which device to copmute on. Default is the first device (indexing from 0).
* -P changes the port that the miner makes API calls to. Use this if you configured Sia to be on a port other than the default.

 
For example, if you wanted to run the program at 64x intensity on device 2, you would call `./gpu-miner -I 22 -d 1`

## Notes
*    Each Sia block takes about 10 minutes to mine.
*    Once a block is mined, Sia waits for 144 confirmation blocks before the reward is added to your wallet, which takes about 24 hours.
*    Sia currently doesn't have any mining pools. A p2pool portal is under development, and can be expected to be ready by the end of the summer.
