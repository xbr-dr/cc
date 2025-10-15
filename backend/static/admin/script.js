// Helper function to upload files
async function uploadFiles(files, endpoint, msgElem) {
  if (files.length === 0) {
    msgElem.textContent = "Please select files first.";
    msgElem.style.color = "red";
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  msgElem.textContent = "Uploading...";
  msgElem.style.color = "black";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (response.ok && data.status === "success") {
      msgElem.textContent = `Upload successful! Files uploaded: ${data.files_uploaded}. Added items: ${data.locations_added ?? data.docs_added ?? "N/A"}`;
      msgElem.style.color = "green";
    } else {
      msgElem.textContent = "Upload failed. Check backend.";
      msgElem.style.color = "red";
    }
  } catch (e) {
    msgElem.textContent = "Error connecting to backend.";
    msgElem.style.color = "red";
    console.error(e);
  }
}

// Upload Locations
const locationFilesInput = document.getElementById("locationFiles");
const uploadLocationsBtn = document.getElementById("uploadLocationsBtn");
const uploadLocationsMsg = document.getElementById("uploadLocationsMsg");

uploadLocationsBtn.onclick = () => {
  uploadFiles(locationFilesInput.files, "http://127.0.0.1:5000/admin/upload_locations", uploadLocationsMsg);
};

// Upload Documents
const docFilesInput = document.getElementById("docFiles");
const uploadDocsBtn = document.getElementById("uploadDocsBtn");
const uploadDocsMsg = document.getElementById("uploadDocsMsg");

uploadDocsBtn.onclick = () => {
  uploadFiles(docFilesInput.files, "http://127.0.0.1:5000/admin/upload_documents", uploadDocsMsg);
};

// Reset Locations with confirmation
const resetLocationsBtn = document.getElementById("resetLocationsBtn");
resetLocationsBtn.onclick = async () => {
  if (!confirm("Are you sure you want to RESET ALL LOCATIONS? This action cannot be undone.")) return;

  try {
    const res = await fetch("http://127.0.0.1:5000/admin/reset_locations", {
      method: "POST"
    });
    const data = await res.json();
    alert(data.message ?? "Locations reset.");
    uploadLocationsMsg.textContent = "";
  } catch (e) {
    alert("Error resetting locations.");
    console.error(e);
  }
};

// Reset Documents with confirmation
const resetDocsBtn = document.getElementById("resetDocsBtn");
resetDocsBtn.onclick = async () => {
  if (!confirm("Are you sure you want to RESET ALL DOCUMENTS? This action cannot be undone.")) return;

  try {
    const res = await fetch("http://127.0.0.1:5000/admin/reset_documents", {
      method: "POST"
    });
    const data = await res.json();
    alert(data.message ?? "Documents reset.");
    uploadDocsMsg.textContent = "";
  } catch (e) {
    alert("Error resetting documents.");
    console.error(e);
  }
};

// Export Analytics
const exportAnalyticsBtn = document.getElementById("exportAnalyticsBtn");
exportAnalyticsBtn.onclick = async () => {
  try {
    const res = await fetch("http://127.0.0.1:5000/admin/export_analytics");
    if (!res.ok) throw new Error("Failed to fetch analytics");

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "user_session_analytics.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (e) {
    alert("Error exporting analytics.");
    console.error(e);
  }
};
