// Tab switching
const btnNavigate = document.getElementById("btnNavigate");
const btnChat = document.getElementById("btnChat");
const sectionNavigate = document.getElementById("navigate");
const sectionChat = document.getElementById("chat");

btnNavigate.onclick = () => {
  btnNavigate.classList.add("active");
  btnChat.classList.remove("active");
  sectionNavigate.classList.add("active");
  sectionChat.classList.remove("active");

  // Fix: Force Leaflet map to re-render when showing Navigate tab
  if (map) {
    setTimeout(() => {
      map.invalidateSize();
    }, 100);
  }
};

btnChat.onclick = () => {
  btnChat.classList.add("active");
  btnNavigate.classList.remove("active");
  sectionChat.classList.add("active");
  sectionNavigate.classList.remove("active");
};

// Globals
let locations = [];
let map, marker;
let baseLayers, returnButtonControl;

// Fetch locations from backend
async function loadLocations() {
  try {
    const res = await fetch("http://127.0.0.1:5000/user/locations");
    locations = await res.json();

    // Sort alphabetically by name (case-insensitive)
    locations.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));

    const select = document.getElementById("locationSelect");
    select.innerHTML = "";

    locations.forEach((loc, i) => {
      const option = document.createElement("option");
      option.value = i;
      option.textContent = loc.name;
      select.appendChild(option);
    });

    if (locations.length > 0) {
      select.selectedIndex = 0;
      initMap();
      updateLocationDisplay();
    } else {
      document.getElementById("locationDetails").textContent = "No locations available.";
    }
  } catch (e) {
    document.getElementById("locationDetails").textContent = "Failed to load locations.";
    console.error(e);
  }
}

function updateLocationDisplay() {
  const select = document.getElementById("locationSelect");
  const loc = locations[select.value];
  if (!loc) return;

  const detailsDiv = document.getElementById("locationDetails");
  detailsDiv.textContent = `${loc.name}\n${loc.details}\nLatitude: ${loc.lat}\nLongitude: ${loc.lon}`;
}


function updateLocationDisplay() {
  const select = document.getElementById("locationSelect");
  const loc = locations[select.value];
  if (!loc) return;

  const detailsDiv = document.getElementById("locationDetails");
  detailsDiv.textContent = `${loc.name}\n${loc.details}\nLatitude: ${loc.lat}\nLongitude: ${loc.lon}`;

  // Update map marker and center map on location
  if (marker) {
    marker.setLatLng([loc.lat, loc.lon]);
    marker.setTooltipContent(loc.name);
  } else if (map) {
    marker = L.marker([loc.lat, loc.lon]).addTo(map);
    marker.bindTooltip(loc.name, { permanent: true, direction: "bottom", offset: [0, 10] }).openTooltip();
  }
  map.setView([loc.lat, loc.lon], 17);
}

// Initialize Leaflet map with multiple base layers and return button
function initMap() {
  if (!map) {
    const loc = locations[0];

    // Base Layers
    
    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '<a href="https://instagram.com/Xbr_Dr" target="_blank" rel="noopener noreferrer">@Xbr_Dr</a> |CampusGPT |© OSM'
    });

    const satellite = L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains:['mt0','mt1','mt2','mt3'],
      attribution: '<a href="https://instagram.com/Xbr_Dr" target="_blank" rel="noopener noreferrer">@Xbr_Dr</a> | CampusGPT | © Google Satellite'
    });


    const terrain = L.tileLayer('https://{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains:['mt0','mt1','mt2','mt3'],
      attribution: '<a href="https://instagram.com/Xbr_Dr" target="_blank" rel="noopener noreferrer">@Xbr_Dr</a> |CampusGPT |© Google Terrain'
    });

    baseLayers = {
      "OpenStreetMap": osm,
      "Satellite": satellite,
      "Terrain": terrain
    };

    map = L.map('map', {
      center: [loc.lat, loc.lon],
      zoom: 16,
      layers: [osm]
    });

    // Add marker
    marker = L.marker([loc.lat, loc.lon]).addTo(map);
    marker.bindTooltip(loc.name, { permanent: true, direction: 'bottom', offset: [0, 10] }).openTooltip();

    // Add layer control to map
    L.control.layers(baseLayers).addTo(map);

    // Add Return to Marker button as custom control
    addReturnToMarkerControl(loc.lat, loc.lon);
  }
}

// Custom Leaflet Control: Return to Marker Button
function addReturnToMarkerControl(lat, lon) {
  if (returnButtonControl) {
    map.removeControl(returnButtonControl);
  }
  returnButtonControl = L.Control.extend({
    options: { position: 'topleft' },
    onAdd: function (map) {
      const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-custom');
      container.style.backgroundColor = 'white';
      container.style.width = '34px';
      container.style.height = '34px';
      container.style.cursor = 'pointer';
      container.style.display = 'flex';
      container.style.alignItems = 'center';
      container.style.justifyContent = 'center';
      container.title = 'Return to Marker';

      container.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#4a90e2" viewBox="0 0 24 24"><path d="M12 2L3 21h18L12 2zm0 3.75L17.53 19H6.47L12 5.75zM11 10v6h2v-6h-2z"/></svg>`;

      container.onclick = () => {
        map.setView([lat, lon], 18);
      };

      return container;
    }
  });
  returnButtonControl = new returnButtonControl();
  returnButtonControl.addTo(map);
}

// When user selects a location from dropdown:
document.getElementById("locationSelect").addEventListener("change", () => {
  updateLocationDisplay();

  // Update Return to Marker button with new coords
  const loc = locations[document.getElementById("locationSelect").value];
  if (loc && map) {
    addReturnToMarkerControl(loc.lat, loc.lon);
  }
});

// Navigate button opens Google Maps navigation
document.getElementById("navigateBtn").addEventListener("click", () => {
  const loc = locations[document.getElementById("locationSelect").value];
  if (!loc) return;
  const googleMapsUrl = `https://www.google.com/maps/dir/?api=1&destination=${loc.lat},${loc.lon}`;
  window.open(googleMapsUrl, "_blank");
});

// Chat functionality
const chatLog = document.getElementById("chatLog");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

// Maintain full chat history (with roles)
const chatHistory = [];

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender === "user" ? "userMsg" : "botMsg"}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// Show welcome message on load
function addWelcomeMessage() {
  const welcome = "👋 Welcome to CampusGPT! Chat here or click 'Navigate' to explore the campus.";
  addMessage(welcome, "bot");
  chatHistory.push({ role: "assistant", content: welcome });
}

sendBtn.onclick = async () => {
  const message = chatInput.value.trim();
  if (!message) return;

  // Add user message to chat log and history
  addMessage(message, "user");
  chatHistory.push({ role: "user", content: message });

  chatInput.value = "";

  try {
    const res = await fetch("http://127.0.0.1:5000/user/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: chatHistory }), // send full chat history
    });
    const data = await res.json();

    // Add bot reply to chat log and history
    addMessage(data.reply, "bot");
    chatHistory.push({ role: "assistant", content: data.reply });
  } catch (e) {
    addMessage("Sorry, something went wrong.", "bot");
    console.error(e);
  }
};

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault(); // Prevent newline in input
    sendBtn.click(); // Trigger send button click
  }
});

// Load locations and add welcome message on page load
window.onload = () => {
  loadLocations();
  addWelcomeMessage();
};
