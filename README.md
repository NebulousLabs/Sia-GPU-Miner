# Sia-GPU-Miner
A GPU miner plugin designed for mining siacoins. Supports Windows, Mac, and Linux. Unzip this file into your plugin directory for Sia.
E.g. C:\Users\John\Downloads\Sia-UI-v0.4.6-beta-win64\Sia-UI-v0.4.4-beta-win64\resources\app\plugins

![alt text](https://i.imgur.com/s2YMbRa.png "This is awesome!")


## How to Use
1) Open Sia-UI

2) Unlock Wallet

3) Click the mining tab and start mining.

## Configuration
*  You can set the intensity with the provided field. On each GPU call, the GPU will do 2^I hashes. The
  default value is low to prevent certain GPUs from crashing immediately at
  startup. Most cards will benefit substantially from increasing the value. The
  default is 20, but recommended (for most cards) is 20-28.

## Multiple GPUs
Not currently supported. The miner will select the first card by default.

## Notes
*    Each Sia block takes about a week or so to mine.
*    Once a block is mined, Sia waits for 144 confirmation blocks before the
	 reward is added to your wallet, which takes about 24 hours.
*    Sia currently doesn't have any mining pools.
