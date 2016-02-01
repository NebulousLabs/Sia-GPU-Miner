// buttons.js
//
// Bindings for buttons in html view.
//
//
// October 22, 2015
// John Drogo 
//

//What happens on 400/500 errors reported by the miner?

const Process = require("child_process").spawn
const path = require("path")
const os = require("os")

basedir = window.location.pathname

var elByID = function (text){ return document.getElementById(text) }
var minerstatus = "idle"
var blocksmined = 0
var hashrate = 0

var miner
var minerfile = ""

var platform = os.platform()
switch (platform){
    case "linux":
        minerfile = "sia-gpu-miner-Linux"
        break;

    case "darwin":
        minerfile = "sia-gpu-miner-Mac"
        break

    case "win32":
        minerfile = "sia-gpu-miner-Windows.exe"
        //Remove nonsense leading slash for Windows.
        basedir = basedir.slice(1)
        break;

    default:
        IPC.sendToHost("notification", "Invalid OS detected. Please use Linux, Mac, or Windows with this plugin.", "error")
        elByID("toggleminer").innerHTML = "Invalid OS"
        break;
}

if (os.arch() != "x64"){
        IPC.sendToHost("notification", "Invalid arch detected. Please use a 64bit processor.", "error")
        elByID("toggleminer").innerHTML = "Invalid Arch"
        minerfile = ""
}


function minerMessage(text){
    //elByID("mineroutput").innerHTML = text + elByID("mineroutput").innerHTML
}


function minerUpdateStatus(){
    //Update status text.
    switch (minerstatus){
        case "idle":
            elByID("minerstatus").innerHTML = "Miner is Idle"
            elByID("toggleminer").innerHTML = "Start GPU Miner!"
            elByID("gpublocks").innerHTML = "Blocks Mined: " + blocksmined
            elByID("hashrate").innerHTML = "Hash Rate: " + hashrate + " MHz"
            elByID("hashratehuge").innerHTML = "Miner is Idle"
            break;

        case "loading":
            elByID("minerstatus").innerHTML = "Miner is Loading"
            elByID("toggleminer").innerHTML = "Miner is Loading"
            elByID("gpublocks").innerHTML = "Loading..."
            elByID("hashrate").innerHTML = "Loading..."
            break

        default:
            elByID("minerstatus").innerHTML = "Miner is Mining"
            elByID("toggleminer").innerHTML = "Stop GPU Miner"
            elByID("gpublocks").innerHTML = "Blocks Mined: " + blocksmined
            elByID("hashrate").innerHTML = "Hash Rate: " + hashrate + " MHz"
            elByID("hashratehuge").innerHTML = hashrate + " MHz"
            break
    }
}



elByID("toggleminer").onclick = function (){
    if (!minerfile){
        IPC.sendToHost("notification", "Invalid OS detected. Please use 64 bit Linux, Mac, or Windows with this plugin.", "error")
        return
    }

    if (minerstatus == "idle"){
        if (elByID("lock").innerHTML != "Unlocked"){
            IPC.sendToHost("notification", "Please unlock your wallet before starting the miner.", "error")
            return
        }


        //Launch the miner!
        intensity = Number(elByID("intensity").value)
        if (intensity < 16 || intensity > 32){
            IPC.sendToHost("notification", "The Intensity Value must be between 16 and 32.", "error")
            return
        }

        miner = Process(
            path.resolve(basedir, "../assets/"+minerfile),
                [ "-I", intensity],{ 
                stdio: [ "ignore", "pipe", "pipe" ],
                cwd: path.resolve(basedir, "../assets")
        })
        minerstatus = "loading"
        minerUpdateStatus()
        IPC.sendToHost('notification',
            "The GPU miner has started with intensity " + intensity  + "!",
        "success");


        miner.stdout.on('data', function (data) {
            console.log('stdout: ' + data);
            minerMessage(data)
           
            minerstatus = "active" 
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
            IPC.sendToHost("notification", "GPU Mining Error: " + data, "error")
            minerMessage(data)
            //minerUpateStatus()
        });
        
        miner.on('exit', function (code) {
            IPC.sendToHost('notification', "The GPU miner has stopped.", "exit");
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
