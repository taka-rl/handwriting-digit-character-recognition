
// Canvas functionality
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.lineWidth = 10;
ctx.lineCap = 'round';
let isDrawing = false;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(event){
  isDrawing = true;
  ctx.beginPath();
  const { offsetX, offsetY } = getEventPosition(event);
  ctx.moveTo(offsetX, offsetY);
}

function draw(event){
  if (!isDrawing) return;
  const { offsetX, offsetY } = getEventPosition(event);
  ctx.lineTo(offsetX, offsetY);
  ctx.stroke();
}

function stopDrawing(){
  isDrawing = false;
}

function clearCanvas(){
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function submitDrawing(){
  const dataURL = canvas.toDataURL('image/png');
  fetch('/submit', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: dataURL })
  })
  .then(response => response.json())
  .then(data => {
    if(data.error){
      alert(`Error: ${data.error}`);
      return;
    }
    updateResults(data);
  })
  .catch(error => console.error('Error:', error));
}

function getEventPosition(event){
  if(event.touches){
    const touch = event.touches[0];
    return { offsetX: touch.clientX - canvas.offsetLeft, offsetY: touch.clientY - canvas.offsetTop };
  }else{
    return { offsetX: event.offsetX, offsetY: event.offsetY };
  }
}

function updateResults(data){
  const predictionElement = document.getElementById('prediction');
  const confidenceElement = document.getElementById('confidence');
  const chartContainer = document.getElementById('chartContainer');

  predictionElement.textContent = data.prediction;
  confidenceElement.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

  // Clear existing chart
  chartContainer.innerHTML = '';

  const maxProbability = Math.max(...data.probabilities);
  data.probabilities.forEach((prob, index) => {
    const bar = document.createElement('div');
    bar.className = 'chart-bar';
    bar.style.height = `${(prob / maxProbability) * 100}%`;
    bar.style.left = `${index * 30}px`;
    if(index === data.prediction){
      bar.classList.add('activ');
    }
    bar.textContent = `${(prob * 100).toFixed(1)}%`;
    chartContainer.appendChild(bar);
    });
}