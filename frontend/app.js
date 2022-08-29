import { Color } from 'three';
import { IfcViewerAPI } from 'web-ifc-viewer';


document.getElementById("model-description").value = "A photograph of my office";
document.getElementById("context-description").value = "in the Amazonian rainforest";
document.getElementById("style-description").value = "Studio photography";

const fileInput = document.getElementById("file-input");
const fileInputButton = document.getElementById("file-input-button");
const ifcExamplesButton = document.getElementById("load-examples");
const ifcExamplesGroup = document.getElementById("ifc-examples-group");
const ifcExample1Button = document.getElementById("ifc-example1");
const ifcExample2Button = document.getElementById("ifc-example2");
const ifcExample3Button = document.getElementById("ifc-example3");
const ifcExample4Button = document.getElementById("ifc-example4");
const ifcExample5Button = document.getElementById("ifc-example5");
const generateButton = document.getElementById("generate-button");
const generatingBlock = document.getElementById("generating-block");
const generatingText = document.getElementById("generating-txt");
const timerRef = document.getElementById("timerDisplay");
const container = document.getElementById("viewer-container");
const unloadedMessage = document.getElementById("unloaded-message");
const loadingAnimation = document.getElementById("loading-animation");

let viewer = null;

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
    fetchWithTimeout("https://cvillagrasa.ddns.net:5000/receiver",
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
            if(timeElapsed !== null){
                clearInterval(timeElapsed);
                [milliseconds, seconds, minutes] = [0, 0, 0];
                timeElapsed = null;
            }
            timerRef.innerHTML = "UNABLE TO PROCESS THE REQUEST"
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
    if (viewer !== null) {
        await viewer.dispose();
    }
    viewer = new IfcViewerAPI({ container, backgroundColor: new Color(0xffffff) });
    ifcExamplesGroup.setAttribute("hidden", "");
    unloadedMessage.style.display = "none";
    loadingAnimation.style.display = "block";
    await viewer.IFC.setWasmPath("./");
    viewer.grid.setGrid();
    viewer.axes.setAxes();
    const model = await viewer.IFC.loadIfcUrl(url);
    await viewer.shadowDropper.renderShadow(model.modelID);
    loadingAnimation.style.display = "none";
}


fileInputButton.onclick = () => fileInput.click();
fileInput.onchange = async (changed) => {
    const ifcURL = URL.createObjectURL(changed.target.files[0]);
    await loadIfc(ifcURL);
};


ifcExamplesButton.onclick = () => {
    if (ifcExamplesGroup.hasAttribute("hidden")) {
        ifcExamplesGroup.removeAttribute("hidden");
    } else {
        ifcExamplesGroup.setAttribute("hidden", "");
    };
};

let exampleFilenames = ["01", "02", "03", "04", "05"];
exampleFilenames = exampleFilenames.map(num => `../sample_files/${num}.ifc`);
const loadExampleFunctions = exampleFilenames.map((path) => {
    return async () => {await loadIfc(path)};
});


ifcExample1Button.onclick = () => loadExampleFunctions[0]();
ifcExample2Button.onclick = () => loadExampleFunctions[1]();
ifcExample3Button.onclick = () => loadExampleFunctions[2]();
ifcExample4Button.onclick = () => loadExampleFunctions[3]();
ifcExample5Button.onclick = () => loadExampleFunctions[4]();


generateButton.onclick = () => {
    if (viewer === null) {
        alert("Please, load an IFC file before!")
        return
    }
    generatingBlock.removeAttribute("hidden");
    generatingText.removeAttribute("hidden");
    if(timeElapsed !== null){
        clearInterval(timeElapsed);
        [milliseconds, seconds, minutes] = [0, 0, 0];
    }
    timeElapsed = setInterval(displayTimer, timerStep);
    inpaintingPipeline();
};
