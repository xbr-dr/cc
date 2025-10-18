// Text formatting function - ADD THIS AT THE TOP
function formatChatMessage(text) {
    if (!text) return '';
    
    // Escape HTML to prevent XSS
    text = text.replace(/&/g, '&amp;')
               .replace(/</g, '&lt;')
               .replace(/>/g, '&gt;');
    
    // Convert **bold** to <strong> (MUST come before single *)
    text = text.replace(/\*\*([^\*]+?)\*\*/g, '<strong>$1</strong>');
    
    // Convert *italic* to <em> (single asterisks only)
    text = text.replace(/(?<!\*)\*([^\*]+?)\*(?!\*)/g, '<em>$1</em>');
    
    // Convert bullet points (•, -, *, ‣)
    text = text.replace(/^[\s]*[•\-\*‣]\s+(.+)$/gm, '<li>$1</li>');
    
    // Convert numbered lists
    text = text.replace(/^[\s]*(\d+)\.\s+(.+)$/gm, '<li>$2</li>');
    
    // Wrap consecutive <li> in <ul>
    text = text.replace(/(<li>.*?<\/li>\s*)+/gs, function(match) {
        return '<ul>' + match + '</ul>';
    });
    
    // Convert paragraphs (double newline)
    let paragraphs = text.split(/\n\s*\n/);
    paragraphs = paragraphs.map(para => {
        para = para.replace(/\n/g, '<br>');
        if (para.includes('<ul>') || para.includes('<ol>')) {
            return para;
        }
        return '<p>' + para + '</p>';
    });
    text = paragraphs.join('');
    
    // Convert URLs to links
    text = text.replace(
        /https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)/g,
        '<a href="$&" target="_blank" rel="noopener noreferrer">$&</a>'
    );
    
    // Convert emails to mailto links
    text = text.replace(
        /([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/g,
        '<a href="mailto:$1">$1</a>'
    );
    
    // Convert phone numbers to tel links
    text = text.replace(
        /(\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4})/g,
        '<a href="tel:$1">$1</a>'
    );
    
    return text;
}



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
    const res = await fetch("/user/locations");
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
      attribution: '<a href="https://github.com/xbr-dr" target="_blank" rel="noopener noreferrer">@Xbr_Dr</a> |CampusGPT |© OSM'
    });

    const satellite = L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains:['mt0','mt1','mt2','mt3'],
      attribution: '<a href="https://github.com/xbr-dr" target="_blank" rel="noopener noreferrer">@Xbr_Dr</a> | CampusGPT | © Google Satellite'
    });

    const terrain = L.tileLayer('https://{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', {
      maxZoom: 20,
      subdomains:['mt0','mt1','mt2','mt3'],
      attribution: '<a href="https://github.com/xbr-dr" target="_blank" rel="noopener noreferrer">@Xbr_Dr</a> |CampusGPT |© Google Terrain'
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

    // Add layer control
    L.control.layers(baseLayers).addTo(map);

    // ✅ Add polygon highlight (always visible on all base layers)
    const polygonCoords = [
      [34.07693442357666, 74.82044936125726],
      [34.07739567744387, 74.82025441484842],
      [34.077862498744, 74.82006175381733],
      [34.07823227007494, 74.81993184977787],
      [34.07833935054218, 74.81985718700484],
      [34.0784198873923, 74.81978901008571],
      [34.07844041731678, 74.81971440791216],
      [34.07846728041613, 74.81966528612156],
      [34.07848856416497, 74.81954204005631],
      [34.07849894463364, 74.81940379824799],
      [34.07848831310557, 74.8192353398189],
      [34.07839400609397, 74.81889521575704],
      [34.07833203447737, 74.81868891364097],
      [34.0782333622525, 74.81841369152252],
      [34.07802412754327, 74.8177008718597],
      [34.07794531097755, 74.81729790848502],
      [34.07797210239354, 74.81728665722136],
      [34.07796616467036, 74.81701247254905],
      [34.07797303410975, 74.81648778536184],
      [34.07797085485774, 74.81577719245142],
      [34.07773260798768, 74.81578228053797],
      [34.07771210515227, 74.81567305021484],
      [34.07760638300436, 74.81566231976721],
      [34.07757131465885, 74.81532674421479],
      [34.07752483117598, 74.81529446157427],
      [34.07751095971062, 74.81503666633893],
      [34.07746716974594, 74.81468314223619],
      [34.07726723390656, 74.81465736662081],
      [34.07696321366854, 74.8146059009765],
      [34.07695557378288, 74.81453513184155],
      [34.07698204000998, 74.81432025195704],
      [34.0769557700089, 74.81429377986015],
      [34.07699077182034, 74.81412555209538],
      [34.07693481870585, 74.81398777890473],
      [34.07680912610917, 74.81382770251069],
      [34.07679861671019, 74.81388830114142],
      [34.07669392977866, 74.81412436068537],
      [34.07650202374226, 74.81454386505968],
      [34.07643425925648, 74.81470453478387],
      [34.07619935973942, 74.81484629059851],
      [34.07604416941525, 74.81493576132162],
      [34.07593374257048, 74.81499305817],
      [34.07562548136259, 74.81515925619168],
      [34.07558312423519, 74.81516789944726],
      [34.07578958419349, 74.8159747202225],
      [34.07588636751581, 74.81626401466656],
      [34.07593408719796, 74.81647882659954],
      [34.07603093562403, 74.81697902560883],
      [34.07619332079385, 74.81761322515239],
      [34.07629106692195, 74.81797268150821],
      [34.07635832075992, 74.81821074047258],
      [34.07647540498265, 74.81871042666258],
      [34.07666590639117, 74.81951504566847],
      [34.07683679574252, 74.82021019809561],
      [34.07693442357666, 74.82044936125726]
    ];

    const highlightedArea = L.polygon(polygonCoords, {
      color: 'red',
      fillColor: '#c57d5aff',
      fillOpacity: 0.3,
      weight: 2
    }).addTo(map);

    highlightedArea.bindTooltip("Campus Area", { permanent: false });

    // Add Return to Marker button
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
  
  // Format bot messages as HTML, keep user messages as text
  if (sender === "bot") {
    div.innerHTML = formatChatMessage(text);
  } else {
    div.textContent = text;
  }
  
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
    const res = await fetch("/user/chat", {
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
