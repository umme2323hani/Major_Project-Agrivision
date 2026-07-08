@app.route('/upload', methods=["POST"])
def upload_file():
    global extracted_data
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        if file.filename.lower().endswith(".pdf"):
            image_files = convert_pdf_to_images(file_path)
            extracted_text = ""
            for image_file in image_files:
                image = preprocess_image(image_file)
                extracted_text += pytesseract.image_to_string(image, config="--oem 3 --psm 6") + "\n"
        else:
            image = preprocess_image(file_path)
            extracted_text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")

        extracted_data = extract_values_from_text(extracted_text)

        return redirect(url_for('index'))
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500
