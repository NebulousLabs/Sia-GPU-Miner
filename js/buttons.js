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
    
    elByID("toggleminer").innerHTML = 
        (minerstatus == "idle" ? "Start GPU Miner!" : "Stop GPU Miner!")

    elByID("gpublocks").innerHTML = "GPU Blocks Mined: " + blocksmined
    elByID("hashrate").innerHTML = "Hash Rate: " + hashrate
}



elByID("toggleminer").onclick = function (){
    if (minerstatus == "idle"){

        //Launch the miner!
        intensity = Number(elByID("intensity").value)
        if (intensity < 16 || intensity > 32){
            IPC.sendToHost("notify", "The Intensity Value must be between 16 and 32.", "error")
            return
        }

        miner = Process(
            path.resolve(basedir, "../assets/sia-gpu-miner"),
                [ "-I", intensity],{ 
                stdio: [ "ignore", "pipe", "pipe" ],
                cwd: path.resolve(basedir, "../assets")
        })
        minerstatus = "active"
        IPC.sendToHost('notify',
            "The GPU miner has started with intensity " + intensity  + "!",
        "start");


        miner.stdout.on('data', function (data) {
            console.log('stdout: ' + data);
            minerMessage(data)
            
            values = String(data).replace("\t", " ").split(" ")
            
            //If we got an update form the miner.
            //You might be asking what is with all the trims.
                //My answer would be what is with all the nulls.
            if (values[0].trim() == "Mining" && values.length == 7){
                hashrate = values[2].trim()
                blocksmined = values[4].trim()
                minerUpdateStatus()
            }
        });
        
        miner.stderr.on('data', function (data) {
            console.log('stderr: ' + data);
            minerMessage(data)
            //minerUpateStatus()
        });
        
        miner.on('exit', function (code) {
            IPC.sendToHost('notify', "The GPU miner has stopped.", "stop");
            console.log("Miner closed.");
            minerstatus = "idle"
            miner = undefined
            hashrate = 0
            minerMessage("Miner stopped.")
            minerUpdateStatus()
        });
    }

    else {
        minerMessage("Sent kill message to miner.")
        miner.kill()
    }
}


process.on("beforeexit", function (){
    if (miner){
        miner.kill()
    }
})
