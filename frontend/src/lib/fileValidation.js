const ALLOWED_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"];

export function validateFiles(files) {
  const invalid = files.filter(
    (f) =>
      !ALLOWED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext)),
  );
  if (invalid.length > 0) {
    return {
      valid: false,
      message: `Unsupported file type: ${invalid.map((f) => f.name).join(", ")}. Only GeoTIFF/TIFF or PNG/JPEG are accepted.`,
    };
  }
  if (files.length > 2) {
    return {
      valid: false,
      message: "Upload at most 2 images (single image, or a pair).",
    };
  }
  return { valid: true, message: null };
}
