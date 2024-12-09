document.getElementById("upload-form").addEventListener("submit", function (event) {
    event.preventDefault();
    
    const fileInput = document.getElementById("image-upload");
    const files = fileInput.files;
    
    if (files.length === 0) {
        alert("Please upload at least one image!");
        return;
    }

    // Clear previous results
    const imagesContainer = document.getElementById("images-container");
    imagesContainer.innerHTML = '';

    // Process each selected image
    Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = function (e) {
            const imageDiv = document.createElement('div');
            imageDiv.classList.add('image-container');
            
            // Create an img element
            const imgElement = document.createElement('img');
            imgElement.src = e.target.result;
            imgElement.style.width = '100%';
            imgElement.style.maxWidth = '400px';
            imgElement.style.marginTop = '20px';
            imageDiv.appendChild(imgElement);

            // Create a div for the result
            const resultDiv = document.createElement('div');
            resultDiv.classList.add('result');
            resultDiv.style.textAlign = 'center'; // Center the result text

            // Prepare the form data for this image
            const formData = new FormData();
            formData.append("image", file);

            // Send the image to the server for prediction
            fetch("/predict", {
                method: "POST",
                body: formData,
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = `<strong style="color: red;">${data.error}</strong>`;
                } else {
                    resultDiv.innerHTML = `<strong>Emotion: ${data.emotion}</strong><br><strong>Confidence: ${(data.confidence * 100).toFixed(2)}%</strong>`;
                }
                imageDiv.appendChild(resultDiv);
            })
            .catch(error => {
                console.error("Error:", error);
                resultDiv.innerHTML = "<strong style='color: red;'>An error occurred!</strong>";
                imageDiv.appendChild(resultDiv);
            });

            // Append the image and result to the container
            imagesContainer.appendChild(imageDiv);
        };
        reader.readAsDataURL(file);
    });
});
