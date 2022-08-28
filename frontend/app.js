import { Color } from 'three';
import { IfcViewerAPI } from 'web-ifc-viewer';


document.getElementById("model-description").value = "A photograph of my office";
document.getElementById("context-description").value = "in the Amazonian rainforest";
document.getElementById("style-description").value = "Studio photography";

const fileInput = document.getElementById("file-input");
const fileInputButton = document.getElementById("file-input-button");
const generateButton = document.getElementById("generate-button");
const generatingBlock = document.getElementById("generating-block");
const generatingText = document.getElementById("generating-txt");
const timerRef = document.getElementById("timerDisplay");
const container = document.getElementById("viewer-container");
const unloadedMessage = document.getElementById("unloaded-message");
const loadingAnimation = document.getElementById("loading-animation");

const viewer = new IfcViewerAPI({ container, backgroundColor: new Color(0xffffff) });

let [milliseconds, seconds, minutes] = [0, 0, 0];
let timeElapsed = null;
let timerStep = 100;


function toggleGrid(bool_state) {
    viewer.grid.grid.visible = bool_state;
    viewer.axes.axes.visible = bool_state;
}


async function fetchWithTimeout(resource, options = {}) {
    const { timeout = 200000 } = options;
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    const response = await fetch(resource, {
        ...options,
        signal: controller.signal
    });
    clearTimeout(id);
    return response;
}


function requestStableDiffusionInpainting(prompt, image){
    fetchWithTimeout("http://127.0.0.1:5000/receiver",
        {
            method: 'POST',
            headers: {
                'Content-type': 'application/json',
                'Accept': 'application/json'
            },
            //
            body: JSON.stringify({
                prompt,
                image
            })}).then(response => {
        if(response.ok){
            function handleResponse(response) {
                return response.json();
            }
            handleResponse(response)
                .then(json => json["imgInpainted"])
                .then(imgInpainted => {
                    document.getElementById("imgInpainted").src = imgInpainted;
                    clearInterval(timeElapsed);
                    generatingText.setAttribute("hidden", "");
                })
        }else{
            alert("something is wrong")
        }
    }).catch((err) => console.error(err));
}


function inpaintingPipeline() {
    const renderer = viewer.context.renderer.renderer;
    toggleGrid(false);
    viewer.context.render();
    const imgData = renderer.domElement.toDataURL();
    toggleGrid(true);
    let data_fields = ["model-description", "context-description", "style-description"];
    data_fields = data_fields.map(field => document.getElementById(field).value);
    const prompt = data_fields[0] + " " + data_fields[1] + ". " + data_fields[2] + ".";
    requestStableDiffusionInpainting(prompt, imgData);
}


function displayTimer(){
    milliseconds+=timerStep;
    if(milliseconds === 1000){
        milliseconds = 0;
        seconds++;
        if(seconds === 60){
            seconds = 0;
            minutes++;
        }
    }

    let m = minutes < 10 ? "0" + minutes : minutes;
    let s = seconds < 10 ? "0" + seconds : seconds;
    let ms = milliseconds / 100
    timerRef.innerHTML = `${m}' ${s}'' ${ms}`;
}


async function loadIfc(url) {
    await viewer.IFC.setWasmPath("node_modules/web-ifc/");
    viewer.grid.setGrid();
    viewer.axes.setAxes();
    const model = await viewer.IFC.loadIfcUrl(url);
    await viewer.shadowDropper.renderShadow(model.modelID);
}


fileInputButton.onclick = () => fileInput.click();
fileInput.onchange = async (changed) => {
    const ifcURL = URL.createObjectURL(changed.target.files[0]);
    unloadedMessage.style.display = "none";
    loadingAnimation.style.display = "block";
    await loadIfc(ifcURL);
    loadingAnimation.style.display = "none";
};


generateButton.onclick = () => {
    generatingBlock.removeAttribute("hidden");
    generatingText.removeAttribute("hidden");
    if(timeElapsed !== null){
        clearInterval(timeElapsed);
        [milliseconds, seconds, minutes] = [0, 0, 0];
    }
    timeElapsed = setInterval(displayTimer, timerStep);
    inpaintingPipeline();
};
