const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const uploadText = document.getElementById("uploadText");
const rateBtn = document.getElementById("rateBtn");
const styleScore = document.getElementById("styleScore");
const recommendationList = document.getElementById("recommendationList");

// Click upload box to open file selector
uploadBox.addEventListener("click", () => fileInput.click());
document.getElementById("browseText").onclick = () => fileInput.click();

// Show image preview when selected
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = () => {
      previewImage.src = reader.result;
      previewImage.style.display = "block";
      uploadText.style.display = "none";
    };
    reader.readAsDataURL(file);
  }
});

// Handle "Rate" button click
rateBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please upload a photo first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Failed to analyze image.");
    }

    const data = await response.json();

    // ✅ Display fashion score
    styleScore.textContent = data.score ? data.score.toFixed(1) : "--";

    // ✅ Clear old recommendations
    recommendationList.innerHTML = "";

    // ✅ Handle structured recommendations
    if (data.recommendations && Array.isArray(data.recommendations)) {
      data.recommendations.forEach(rec => {
        const li = document.createElement("li");

        // Handle both dicts and plain strings
        if (typeof rec === "object" && rec.tip) {
          li.innerHTML = `<strong>${rec.category}</strong>: ${rec.tip}`;

          // Priority-based color
          if (rec.priority === "high") li.style.color = "red";
          else if (rec.priority === "medium") li.style.color = "orange";
          else li.style.color = "green";

          // Optional: emoji to make it more aesthetic 😎
          if (rec.priority === "high") li.prepend("🔥 ");
          else if (rec.priority === "medium") li.prepend("⚡ ");
          else li.prepend("✅ ");
        } else {
          li.textContent = rec;
        }

        recommendationList.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "No recommendations available.";
      recommendationList.appendChild(li);
    }
  } catch (error) {
    console.error("❌ Backend Error:", error);
    alert("Error connecting to backend!");
  }
});
