const CANVAS_SIZE = 280;
const CANVAS_SCALE = .5;

// const walkButton = document.getElementById("walk-button");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const refreshButton = document.getElementById("refresh-button");
const interButton = document.getElementById("lerp-button");


var leArray = new Float32Array(Array.from({length: 100}, () => (Math.random()*2) -1));

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

// Load our model.
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./onnx_model.onnx");

// // Add 'Draw a number here!' to the canvas.
// ctx.lineWidth = 28;
// ctx.lineJoin = "round";
// ctx.font = "28px sans-serif";
// ctx.textAlign = "center";
// ctx.textBaseline = "middle";
// ctx.fillStyle = "#212121";

// Set the line color for the canvas.
ctx.strokeStyle = "#212121";

async function updateCanvas(leArray) {
    const input = new onnx.Tensor(leArray, 'float32', [1,100]);
    await loadingModelPromise;
    const outputMap = await sess.run([input]);
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data;

    var imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    var data = imageData.data;

    for (var i = 0; i < 280*280; i ++) {

        // console.log(Math.floor(i/2800),i%28, );
        row = Math.floor(i/2800);
        column = Math.floor(i/10)%28
        smallcoordinate = row*28+column
        // console.log(row,column,smallcoordinate)

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
    let size = 5;
    for (var x = 0; x<size+1; x++){
        leArray1 = new Float32Array(Array.from({length: 100}, () => (Math.random()*2) -1 ));
        tempArray = new Float32Array(100);

        let speed = 0.02;
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

function task(i,Tensor,delay=0) { 
    setTimeout(function() { 
        updateCanvas(Tensor)
    }, 40*i +delay); 
} 

function addTensor(a,b){
    return a.map((e,i) => e + b[i]);
}

function interpolateTensor(a,b,t){
    return a.map((e,i) => (1-t)*e + t*b[i]);
}

function drawLine(fromX, fromY, toX, toY) {
    // Draws a line from (fromX, fromY) to (toX, toY).
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.closePath();
    ctx.stroke();
    updatePredictions();
  }


function canvasMouseDown(event) {
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false;
  }
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  // To draw a dot on the mouse down event, we set laxtX and lastY to be
  // slightly offset from x and y, and then we call `canvasMouseMove(event)`,
  // which draws a line from (laxtX, lastY) to (x, y) that shows up as a
  // dot because the difference between those points is so small. However,
  // if the points were the same, nothing would be drawn, which is why the
  // 0.001 offset is added.
  lastX = x + 0.001;
  lastY = y + 0.001;
  canvasMouseMove(event);
}

function canvasMouseMove(event) {
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;
  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }
  lastX = x;
  lastY = y;
}

function bodyMouseUp() {
  isMouseDown = false;
}

function bodyMouseOut(event) {

  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
}

// canvas.addEventListener("mousedown", canvasMouseDown);
// canvas.addEventListener("mousemove", canvasMouseMove);
// document.body.addEventListener("mouseup", bodyMouseUp);
// document.body.addEventListener("mouseout", bodyMouseOut);
// walkButton.addEventListener("mousedown", walkCanvas);
refreshButton.addEventListener("mousedown", refreshCanvas);
interButton.addEventListener("mousedown", lerpCanvas);
