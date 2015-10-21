# Sia-GPU-Miner
A GPU miner designed for mining siacoins. This miner runs in a command prompt
and prints your hashrate along side the number of blocks you've mined. Most
cards will see greatly increased hashrates by increasing the value of 'I'
(default is 20, optimal is typically 20-28).

## How to Use
1) Download the latest build from: https://github.com/droghio/Sia-GPU-Miner/releases

2) Make sure you have a recent version of Sia installed and running. Don't forget to unlock your wallet.

3) Run the miner by double clicking the `gpu-miner.exe`. It will mine blocks until killed with Ctrl-C.

## Configuration
You can tweak the miner settings with five command-line arguments: `-I`, `-C`, `-p`, `-d`, and `-P`.
* -I controls the intensity. On each GPU call, the GPU will do 2^I hashes. The
  default value is low to prevent certain GPUs from crashing immediately at
  startup. Most cards will benefit substantially from increasing the value. The
  default is 20, but recommended (for most cards) is 20-28.
* -C controls how frequently calls to siad are made. Reducing it substantially
  could cause instability to the miner. Increasing it will reduce the frequency
  at which the hashrate is updated. If a low 'I' is being used, a high 'C'
  should be used. As a rule of thumb, the hashrate should only be updating a
  few times per second. The default is 30.
* -p allows you to pick a platform. Default is the first platform (indexing
  from 0).
* -d allows you to pick which device to copmute on. Default is the first device
  (indexing from 0).
* -P changes the port that the miner makes API calls to. Use this if you
  configured Sia to be on a port other than the default. Default is 9980.

If you wanted to run the program on platform 0, device 1, with an intensity of
24, you would call `sia-gpu-miner -d 1 -I 24`

## Multiple GPUs
Each instance of the miner can only point to a single GPU. To mine on multiple
GPUs at the same time, you will need to run multiple instances of the miner and
point each at a different gpu. Only one instance of 'siad' needs to be running,
all of the miners can point to it.

It is highly recommended that you greatly increase the value of 'C' when using
multiple miners. As a rule of thumb, the hashrate for each miner should be
updating one time per [numGPUs] seconds. You should not mine with more than 6
cards at a time (per instance of siad).

## Notes
*    Each Sia block takes about a week or so to mine.
*    Once a block is mined, Sia waits for 144 confirmation blocks before the
	 reward is added to your wallet, which takes about 24 hours.
*    Sia currently doesn't have any mining pools.
