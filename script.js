/* ---------- DOM helpers ---------- */
const $ = id => document.getElementById(id);

/* elements */
const localVideo   = $('localVideo');
const remoteVideo  = $('remoteVideo');
const callerCanvas = $('callerCanvas');
const combined     = $('combinedCanvas');
const hostBtn      = $('hostBtn');
const callerBtn    = $('callerBtn');
const peerSection  = $('peerSection');
const peerIdDisp   = $('peerIdDisplay');
const peerIdInput  = $('peerIdInput');
const connectBtn   = $('connectBtn');
const styledCanvas = document.getElementById('styledCanvas');
const styledCtx = styledCanvas.getContext('2d');

let peer, localStream;

/* ---------- camera ---------- */
async function getCamera() {
  if (localStream) return localStream;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    localVideo.srcObject = localStream;
  } catch (e) {
    console.error('getUserMedia', e.name);
  }
  connectBtn.disabled = false;
  return localStream;
}

/* ---------- MediaPipe Selfie-Seg ---------- */
const selfieSeg = new SelfieSegmentation({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}`
});
selfieSeg.setOptions({ modelSelection: 1 });

/* canvas contexts */
const silCtx  = callerCanvas.getContext('2d');
const combCtx = combined.getContext('2d');

/* ---------- draw each result ---------- */
selfieSeg.onResults(res => {
  const W = callerCanvas.width, H = callerCanvas.height;

  silCtx.clearRect(0, 0, W, H);
  silCtx.drawImage(remoteVideo, 0, 0, W, H);
  silCtx.globalCompositeOperation = 'destination-in';
  silCtx.drawImage(res.segmentationMask, 0, 0, W, H);
  silCtx.globalCompositeOperation = 'source-over';

  combCtx.clearRect(0, 0, W, H);
  combCtx.drawImage(localVideo, 0, 0, W, H);
  combCtx.globalCompositeOperation = 'destination-out';
  combCtx.drawImage(res.segmentationMask, 0, 0, W, H);
  combCtx.globalCompositeOperation = 'source-over';
  combCtx.drawImage(callerCanvas, 0, 0, W, H);
});

/* ---------- drive MediaPipe ---------- */
function startSegLoop() {
  (async function loop() {
    if (remoteVideo.readyState >= 2) {
      await selfieSeg.send({ image: remoteVideo });
    }
    requestAnimationFrame(loop);
  })();
}

/* ---------- PeerJS plumbing ---------- */
const newPeer = () => new Peer({ host: '0.peerjs.com', port: 443, path: '/', secure: true, debug: 2 });

function handleRemoteStream(stream) {
  remoteVideo.srcObject = stream;
  remoteVideo.onloadeddata = startSegLoop;
}

function wireCall(call, role) {
  call.on('stream', handleRemoteStream);
  call.on('error', e => console.error(`[${role}] call`, e.message));
}

/* ---------- HOST ---------- */
hostBtn.onclick = async () => {
  await getCamera();
  peer = newPeer();
  peer.on('open', id => {
    peerIdDisp.textContent = `Share this ID: ${id}`;
    peerSection.style.display = 'block';
  });
  peer.on('call', c => { c.answer(localStream || undefined); wireCall(c, 'HOST'); });
};

/* ---------- CALLER ---------- */
callerBtn.onclick = async () => {
  await getCamera();
  peer = newPeer();
  peer.on('open', () => (peerSection.style.display = 'block'));
  connectBtn.onclick = () => {
    const hostId = peerIdInput.value.trim();
    if (!hostId) return alert('Enter host ID first');
    wireCall(peer.call(hostId, localStream || undefined), 'CALLER');
  };
};

/* ---------- Styling API Call ---------- */
async function sendForStyling() {
  // Create status element if it doesn't exist
  let statusElem = document.getElementById('styleStatus');
  if (!statusElem) {
    statusElem = document.createElement('p');
    statusElem.id = 'styleStatus';
    styledCanvas.parentNode.insertBefore(statusElem, styledCanvas.nextSibling);
  }
  
  statusElem.textContent = 'Checking canvas data...';
  
  const dataURL = combined.toDataURL('image/png');

  if (!dataURL || dataURL.length < 100) {
    console.error('Combined canvas is empty or invalid.');
    statusElem.textContent = 'Error: Canvas is empty. Make sure both videos are connected first.';
    return;
  }
  
  statusElem.textContent = 'Sending image to server...';
  console.log('Canvas data length:', dataURL.length);

  try {
    // Use 127.0.0.1 instead of localhost to match server
    const serverUrl = 'https://127.0.0.1:5000/style';
    
    // Log connection attempt
    console.log(`Attempting to connect to ${serverUrl}`);
    
    const res = await fetch(serverUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL }),
      mode: 'cors'
    });

    console.log('Response status:', res.status);

    if (!res.ok) {
      throw new Error(`Server error: ${res.status} ${res.statusText}`);
    }

    console.log('Response received, parsing JSON...');
    const data = await res.json();
    console.log('JSON parsed successfully');

    if (!data.image || data.image.length < 100) {
      throw new Error('Received invalid image data from server');
    }

    console.log('Received image data length:', data.image.length);

    const img = new Image();
    img.onload = () => {
      console.log('Image loaded successfully, dimensions:', img.width, 'x', img.height);
      styledCanvas.width = img.width;
      styledCanvas.height = img.height;
      styledCtx.drawImage(img, 0, 0);
      statusElem.textContent = 'Styling complete!';
    };
    img.onerror = (e) => {
      console.error('Image load error:', e);
      throw new Error('Failed to load the styled image');
    };
    img.src = data.image;
  } catch (err) {
    console.error('Request failed:', err);
    statusElem.textContent = `Error: ${err.message}`;
    
    if (err.message.includes('certificate') || err.message.includes('ssl') || 
        err.message.includes('trust') || err.name === 'TypeError') {
      statusElem.innerHTML = `Error: HTTPS certificate issue.<br>
        Please open <a href="https://127.0.0.1:5000/test" target="_blank">https://127.0.0.1:5000/test</a> directly 
        in your browser and accept the certificate warning first.`;
    }
  }
}

/* ---------- Add a debugging function ---------- */
function saveCanvas() {
  const dataURL = combined.toDataURL('image/png');
  if (!dataURL || dataURL.length < 100) {
    alert('Canvas is empty or invalid!');
    return;
  }
  
  const newTab = window.open();
  newTab.document.write(`<img src="${dataURL}" alt="Canvas Image">`);
  console.log('Canvas data length:', dataURL.length);
}