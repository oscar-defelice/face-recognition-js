const MODEL_URL = "utils/face-api/weights";
const loader = document.getElementById("loader");
const video = document.getElementById("webcam");
const videoSize = {
  width: 1280,
  height: 720
};

let trainIter = 0;
let maxIterFineTune = 100;
let faceDescriptors = [];
let trainedFace;
let maxDescriptorDistance = 0.7;

// event listener, to run the training and predicting routine at the "play" event.
video.addEventListener("play", async () => {
  const faceOverlay = faceapi.createCanvasFromMedia(video);
  document.body.append(faceOverlay);
  faceOverlay.style = "z-index: 1;position: absolute;top: " + video.offsetTop + ";left: " + video.offsetLeft + ";";
  faceapi.matchDimensions(faceOverlay, videoSize);

  setInterval(async () => {
    if (trainIter < maxIterFineTune) {
      trainIter++;
      console.log("Training step: %s", trainIter);
      // detect the face with the highest score in the image and compute it's landmarks and face descriptor
      const faceDescriptor = await faceapi.detectSingleFace(video).withFaceLandmarks().withFaceDescriptor();
      // if frame does not contains a face throw error.
      if (!faceDescriptor) {
        throw new Error("Face detection training error.")
      }
      // add single frame face to the training set.
      faceDescriptors.push(faceDescriptor.descriptor);
    } else if (trainIter == maxIterFineTune) {
      trainIter++;
      const labelName = document.getElementById("labelName").value;
      trainedFace = new faceapi.LabeledFaceDescriptors(labelName, faceDescriptors); // add labels to faceDescriptors list
      console.log("Trained completed!")
    } else {
      const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors();

      // create FaceMatcher with automatically assigned labels
      // from the detection results for the reference image
      const faceMatcher = new faceapi.FaceMatcher(trainedFace, maxDescriptorDistance);
      const results = detections.map(fd => faceMatcher.findBestMatch(fd.descriptor));
      const resizedFaces = faceapi.resizeResults(detections, videoSize);
      faceOverlay.getContext('2d').clearRect(0, 0, faceOverlay.width, faceOverlay.height);
      faceapi.draw.drawDetections(faceOverlay, resizedFaces);
      //faceapi.draw.drawFaceLandmarks(faceOverlay, resizedFaces);
      //faceapi.draw.drawFaceDescriptors(faceOverlay, resizedFaces);

      results.forEach((bestMatch, i) => {
        const box = detections[i].detection.box;
        const label = bestMatch.toString();
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: label
        });
        drawBox.draw(faceOverlay);
      });

    }
  }, 10) // set the face recognition api call every 10 ms
});

// initWebcam function.
const initWebcam = async () => {
  const config = {
    audio: false,
    video: videoSize
  };
  const mediaStream = await navigator.mediaDevices.getUserMedia(config);

  video.srcObject = mediaStream;
  video.onloadedmetadata = async () => video.play();
};

let loadedPercentage = 1

// set the progress bar to show the model loading.
const onModelLoaded = () => {
  loadedPercentage += 33
  loader.style = "width: " + loadedPercentage + "%;"
}

// change the bar message once the model has been loaded.
const modelLoaded = () => {
  if (loadedPercentage == 100) {
    loader.innerText = "Model Loaded!";
  }
};

// main function to load models, init webcam and run the routines.
const main = async () => {
  await faceapi.loadSsdMobilenetv1Model(MODEL_URL).then(onModelLoaded);
  await faceapi.loadFaceLandmarkModel(MODEL_URL).then(onModelLoaded);
  await faceapi.loadFaceRecognitionModel(MODEL_URL).then(onModelLoaded);
  initWebcam();
  setInterval(modelLoaded(), 100000);
};

loader.style = "width: " + loadedPercentage + "%;";
main();
