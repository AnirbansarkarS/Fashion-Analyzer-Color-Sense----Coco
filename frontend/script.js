const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const uploadText = document.getElementById("uploadText");
const rateBtn = document.getElementById("rateBtn");
const styleScore = document.getElementById("styleScore");
const recommendationList = document.getElementById("recommendationList");

uploadBox.addEventListener("click", () => fileInput.click());
document.getElementById("browseText").onclick = () => fileInput.click();

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

rateBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please upload a photo first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    styleScore.textContent = data.score ? data.score.toFixed(1) : "--";

    recommendationList.innerHTML = "";
    if (data.recommendations) {
      data.recommendations.forEach(rec => {
        const li = document.createElement("li");
        li.textContent = rec;
        recommendationList.appendChild(li);
      });
    }
  } catch (error) {
    alert("Error connecting to backend!");
    console.error(error);
  }
});
