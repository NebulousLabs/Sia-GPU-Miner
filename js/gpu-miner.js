'use strict';

/*
// Library for communicating with Sia-UI
const IPC = require('electron').ipcRenderer;//require('ipc');
// Library for arbitrary precision in numbers
const BigNumber = require('../../js/bignumber.min.js');
// Ensure precision
BigNumber.config({ DECIMAL_PLACES: 24 });
BigNumber.config({ EXPONENTIAL_AT: 1e+9 });
// Keeps track of if the view is shown
var updating;

// Make API calls, sending a channel name to listen for responses
function update() {
    Siad.call('/wallet', updateWallet);
    Siad.call('/gateway', updatePeers);
    Siad.call('/consensus', updateHeight);
    updating = setTimeout(update, 5000);
}

// Updates element text
function updateField(err, caption, value, elementID) {
	if (err) {
		IPC.sendToHost('notification', 'API call errored!', 'error');
	} else if (value === null) {
		IPC.sendToHost('notification', 'API result seems to be null!', 'error');
	} else {
		document.getElementById(elementID).innerHTML = caption + value;
	}
}

// Convert to Siacoin
function formatSiacoin(hastings) {
	// TODO: JS automatically loses precision when taking numbers from the API.
	// This deals with that imperfectly, rounding to nearest hasting
	var number = new BigNumber(hastings);
	var ConversionFactor = new BigNumber(10).pow(24);
	// Display two digits of Siacoin
	var display = number.dividedBy(ConversionFactor).round(2) + ' S';
	return display;
}

// Called by the UI upon showing
function start() {
	// DEVTOOL: uncomment to bring up devtools on plugin view
	// IPC.sendToHost('devtools');
	
	// Call the API
	update();
}

// Called by the UI upon transitioning away from this view
function stop() {
	clearTimeout(updating);
}

// Ask UI to show tooltip bubble
function tooltip(message, element) {
	var rect = element.getBoundingClientRect();
	IPC.sendToHost('tooltip', message, {
		top: rect.top,
		bottom: rect.bottom,
		left: rect.left,
		right: rect.right,
		height: rect.height,
		width: rect.width,
		length: rect.length,
	});
}

// Define IPC listeners and update DOM per call
IPC.on('wallet-update', function(err, result) {
	if(!result){
		return;
	}

	var unlocked = result.unlocked;
	var encrypted = result.encrypted;
	if (!encrypted) {
		updateField(err, 'New Wallet', '', 'lock');
	} else if (unlocked) {
		updateField(err, 'Unlocked', '', 'lock');
	} else {
		updateField(err, 'Locked', '', 'lock');
	}

	var bal = formatSiacoin(result.confirmedsiacoinbalance);
	updateField(err, 'Balance: ', unlocked ? bal : '---', 'balance');
});
//IPC.on('peers-update', function(err, result) {
//	var value = result !== null ? result.Peers.length : null;
//	updateField(err, 'Peers: ', value, 'peers');
//});
//IPC.on('height-update', function(err, result) {
//	var value = result !== null ? result.height : null;
//	updateField(err, 'Block Height: ', value, 'height');
//});
*/





'use strict';

// Library for communicating with Sia-UI
const IPCRenderer = require('electron').ipcRenderer;
const IPC = IPCRenderer;
// Library for arbitrary precision in numbers
const BigNumber = require('bignumber.js');
// Siad wrapper
const Siad = require('sia.js');

// Ensure precision
BigNumber.config({ DECIMAL_PLACES: 24 });
BigNumber.config({ EXPONENTIAL_AT: 1e+9 });

// Make sure Siad settings are in sync with the rest of the UI's
var settings = IPCRenderer.sendSync('config', 'siad');
Siad.configure(settings);

// Keeps track of if the view is shown
var updating;

// DEVTOOL: uncomment to bring up devtools on plugin view
// IPCRenderer.sendToHost('devtools');

// Returns if API call has an error or null result
function errored(err, result) {
    if (err) {
        console.error(err);
        IPCRenderer.sendToHost('notification', err.toString(), 'error');
        return true;
    } else if (!result) {
        IPCRenderer.sendToHost('notification', 'API result not found!', 'error');
        return true;
    }
    return false;
}

// Convert to Siacoin
function formatSiacoin(hastings) {
    var number = new BigNumber(hastings);
    var ConversionFactor = new BigNumber(10).pow(24);
    // Display two digits of Siacoin
    var display = number.dividedBy(ConversionFactor).round(2) + ' S';
    return display;
}

// Update wallet balance and lock status from call result
function updateWallet(err, result) {
    if (errored(err, result)) {
        return;
    }

    var unlocked = result.unlocked;
    var unencrypted = !result.encrypted;

    var lockText = unencrypted ? 'New Wallet' : unlocked ? 'Unlocked' : 'Locked';
    document.getElementById('lock').innerText = lockText;

    var bal = unlocked ? formatSiacoin(result.confirmedsiacoinbalance) : '--';
    document.getElementById('balance').innerText = 'Balance: ' + bal;
}

// Update peer count from call result
function updatePeers(err, result) {
    if (errored(err, result)) {
        return;
    }
    document.getElementById('peers').innerText = 'Peers: ' + result.peers.length;
}

// Update block height from call result
function updateHeight(err, result) {
    if (errored(err, result)) {
        return;
    }
    document.getElementById('height').innerText = 'Block Height: ' + result.height;
}

// Make API calls, sending a channel name to listen for responses
function update() {
    Siad.call('/wallet', updateWallet);
    Siad.call('/gateway', updatePeers);
    Siad.call('/consensus', updateHeight);
    updating = setTimeout(update, 5000);
}

// Called upon showing
IPCRenderer.on('shown', update);
// Called upon transitioning away from this view
IPCRenderer.on('hidden', function() {
    clearTimeout(updating);
});
