const CANVAS_SIZE = 280;
const CANVAS_SCALE = .5;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const refreshButton = document.getElementById("refresh-button");
const interButton = document.getElementById("lerp-button");

var lerpRunning = false;
// speed = 0.04 for 1 second changes
var speed = 0.02;
var size = 5;

var leArray = new Float32Array(Array.from({length: 100}, () => (Math.random()*2) -1));

const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./onnx_model.onnx");

async function updateCanvas(leArray) {
    const input = new onnx.Tensor(leArray, 'float32', [1,100]);
    await loadingModelPromise;
    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data;

    var imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    var data = imageData.data;

    for (var i = 0; i < 280*280; i ++) {
        row = Math.floor(i/2800);
        column = Math.floor(i/10)%28
        smallcoordinate = row*28+column

        value = (predictions[smallcoordinate] + 1) * 127.5;

        data[4*i]       = value;
        data[4*i+1]     = value; 
        data[4*i+2]     = value;
        data[4*i+3]     = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

async function refreshCanvas() {
    leArray = new Float32Array(Array.from({length: 100}, () => (Math.random()*2) -1 ));
    updateCanvas(leArray)
}

async function walkCanvas(){
    let i = 0; 
    while (i < 180) { 
        leArray1 = new Float32Array(Array.from({length: 100}, () => ((Math.random()*2) -1)*Math.random()*Math.random()/5 ));
        leArray = addTensor(leArray,leArray1);
        task(i,leArray);
        i++; 
    }
} 

async function lerpCanvas(){
    if ( !lerpRunning ) {
      lerpRunning = true;
      for (var x = 0; x<size; x++){
        leArray1 = new Float32Array(Array.from({length: 100}, () => (Math.random()*2) -1 ));
        tempArray = new Float32Array(100);

        let t = 0.00;
        let i = 0;
        while ( t<= 1){
            t = t + speed;
            tempArray = interpolateTensor(leArray,leArray1,t);
            task(i,tempArray,x*40* (1/speed));
            i++;
        }
        leArray = tempArray;
      }
    }

}

function task(i,Tensor,delay=0) { 
    setTimeout(function() { 
        if ((1/speed)*size *40-40== 40*i +delay){
          lerpRunning = false;
        }
        updateCanvas(Tensor)
    }, 40*i +delay); 
} 

function addTensor(a,b){
    return a.map((e,i) => e + b[i]);
}

function interpolateTensor(a,b,t){
    return a.map((e,i) => (1-t)*e + t*b[i]);
}

refreshButton.addEventListener("mousedown", refreshCanvas);
interButton.addEventListener("mousedown", lerpCanvas);