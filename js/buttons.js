// buttons.js
//
// Bindings for buttons in html view.
//
//
// October 22, 2015
// John Drogo 
//

const Process = require("child_process").spawn
const path = require("path")

basedir = window.location.pathname

var elByID = function (text){ return document.getElementById(text) }
var minerstatus = "idle"
var blocksmined = 0
var hashrate = 0

var miner

function minerMessage(text){
    elByID("mineroutput").innerHTML = text + elByID("mineroutput").innerHTML
}

function minerUpdateStatus(){
    //Update status text.
    elByID("minerstatus").innerHTML = 
        (minerstatus == "idle" ? "Miner is Idle" : "Miner is Mining")

    
    elByID("gpublocks").innerHTML = "GPU Blocks Mined: " + blocksmined
    elByID("hashrate").innerHTML = "Hash Rate: " + hashrate
}

elByID("toggleminer").onclick = function (){
    if (minerstatus == "idle"){
        //Launch the miner!
        miner = Process(
        path.resolve(basedir, "../assets/sia-gpu-miner"),{ 
            stdio: [ "ignore", "pipe", "pipe" ],
            cwd: path.resolve(basedir, "../assets")
        })


        miner.stdout.on('data', function (data) {
            console.log('stdout: ' + data);
            minerMessage(data)
            
        });
        
        miner.stderr.on('data', function (data) {
            console.log('stderr: ' + data);
            minerMessage(data)
        });
        
        miner.on('close', function (code) {
            console.log("Miner closed.");
            minerstatus = idle
            miner = undefined
            minerMessage("Miner stopped.")
            minerUpdateStatus()
        });
    }
}
