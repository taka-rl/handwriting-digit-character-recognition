
// Canvas functionality
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const bgColor = '#000000'  // Default: black background
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = '#ffffff';  // Default: white line

// Set initial background color
function setCanvasBackground(color) {
  ctx.fillStyle = color; // Set the background color
  ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill the canvas with the background color
}
setCanvasBackground(bgColor);

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
  setCanvasBackground(bgColor); // Reset the background color
}
function changeDrawingColor(color){
  ctx.strokeStyle = color; // Update the stroke color
}

function submitDigitDrawing(){
  const dataURL = canvas.toDataURL('image/png');
  fetch('/submit-digit', {
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
    updateDigitResults(data);
  })
  .catch(error => console.error('Error:', error));
}

function submitCharacterDrawing(){
  const dataURL = canvas.toDataURL('image/png');
  fetch('/submit-character', {
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
    updateCharacterResults(data);
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

function updateDigitResults(data){
  const probabilities = data.probabilities[0];  // Access the inner array of probabilities
  const predictionElement = document.getElementById('prediction');
  const confidenceElement = document.getElementById('confidence');
  const chartContainer = document.getElementById('chartContainer');

  predictionElement.textContent = data.prediction;
  confidenceElement.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

  // Clear existing chart
  chartContainer.innerHTML = '';

  // Create the distribution bar chart
  probabilities.forEach((prob, index) => {
    const bar = document.createElement('div');
    bar.className = 'chart-bar-digit';
    bar.style.height = `${prob * 100}%`;
    // Highlight the predicted bar
    if(index === data.prediction){
      bar.classList.add('active');
    }
    // Add text label inside the bar
    bar.textContent = `${(prob * 100).toFixed(1)}%`;
    chartContainer.appendChild(bar);
  });
}

function updateCharacterResults(data){
  const upperProbabilities = data.upper_probabilities;  // Uppercase probabilities
  const lowerProbabilities = data.lower_probabilities;  // Lowercase probabilities
  const predictionElement = document.getElementById('prediction');
  const confidenceElement = document.getElementById('confidence');
  const upperChartContainer = document.getElementById('upperChartContainer');
  const lowerChartContainer = document.getElementById('lowerChartContainer');

  predictionElement.textContent = data.prediction;
  confidenceElement.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

  // Clear existing chart
  upperChartContainer.innerHTML = '';
  lowerChartContainer.innerHTML = '';

  // Create the Uppercase distribution bar chart
  upperProbabilities.forEach((prob, index) => {
    const bar = document.createElement('div');
    bar.className = 'chart-bar-character';
    bar.style.height = `${prob * 100}%`;
    // Highlight the predicted bar
    if(index === data.prediction){
      bar.classList.add('active');
    }
    // Add text label inside the bar
    bar.textContent = `${(prob * 100).toFixed(1)}%`;
    upperChartContainer.appendChild(bar);
  });

  // Create the Lowercase distribution bar chart
  lowerProbabilities.forEach((prob, index) => {
    const bar = document.createElement('div');
    bar.className = 'chart-bar-character';
    bar.style.height = `${prob * 100}%`;
    // Highlight the predicted bar
    if(index+26 === data.prediction){
      bar.classList.add('active');
    }
    // Add text label inside the bar
    bar.textContent = `${(prob * 100).toFixed(1)}%`;
    lowerChartContainer.appendChild(bar);
  });
}

function showCorrectionInput() {
    document.getElementById("correctionForm").style.display = "block";
}

function submitFeedback(isCorrect) {
    const dataURL = canvas.toDataURL('image/png');
    const confidenceElement = document.getElementById('confidence').textContent;
    const predictedLabel = document.getElementById("prediction").textContent;

    let correctLabel;
    if (isCorrect) {
        correctLabel = document.getElementById("prediction").textContent; // Use the predicted value
    } else {
        correctLabel = document.getElementById("correctLabel").value; // Get the manually entered label
    }

    fetch('/submit-feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: dataURL,
            predicted_label: predictedLabel,
            confidence: confidenceElement,
            correct_label: correctLabel
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("Feedback submitted successfully!");
        } else {
            alert(`Error: ${data.error}`);
        }
    })
    .catch(error => console.error('Error:', error));
}
