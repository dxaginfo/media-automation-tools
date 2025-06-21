import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import cv2
import numpy as np
from datetime import datetime
import google.cloud.vision as vision

class ValidationError:
    """Class representing a validation error."""
    
    def __init__(self, timecode: str, message: str, severity: str = "error"):
        """
        Initialize a validation error.
        
        Args:
            timecode: Timestamp where the error occurred (e.g., "00:01:23.456")
            message: Description of the error
            severity: Error severity ("error", "warning", "info")
        """
        self.timecode = timecode
        self.message = message
        self.severity = severity
    
    def __str__(self) -> str:
        return f"{self.severity.upper()} at {self.timecode}: {self.message}"


class ValidationResult:
    """Class representing the result of a validation operation."""
    
    def __init__(self):
        """Initialize an empty validation result."""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.info: List[ValidationError] = []
        self.is_valid: bool = True
        self.validation_time: datetime = datetime.now()
    
    def add_error(self, timecode: str, message: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(ValidationError(timecode, message, "error"))
        self.is_valid = False
    
    def add_warning(self, timecode: str, message: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(ValidationError(timecode, message, "warning"))
    
    def add_info(self, timecode: str, message: str) -> None:
        """Add an informational message to the validation result."""
        self.info.append(ValidationError(timecode, message, "info"))
    
    def save_report(self, output_path: str) -> None:
        """
        Save the validation report to a file.
        
        Args:
            output_path: Path where to save the report
        """
        extension = output_path.split('.')[-1].lower()
        
        if extension == 'json':
            self._save_json_report(output_path)
        elif extension == 'html':
            self._save_html_report(output_path)
        elif extension == 'pdf':
            self._save_pdf_report(output_path)
        else:
            raise ValueError(f"Unsupported report format: {extension}")
    
    def _save_json_report(self, output_path: str) -> None:
        """Save the report in JSON format."""
        report = {
            "is_valid": self.is_valid,
            "validation_time": self.validation_time.isoformat(),
            "errors": [{"timecode": e.timecode, "message": e.message, "severity": e.severity} for e in self.errors],
            "warnings": [{"timecode": w.timecode, "message": w.message, "severity": w.severity} for w in self.warnings],
            "info": [{"timecode": i.timecode, "message": i.message, "severity": i.severity} for i in self.info]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _save_html_report(self, output_path: str) -> None:
        """Save the report in HTML format."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scene Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .summary { margin: 20px 0; padding: 10px; border-radius: 5px; }
                .valid { background-color: #dff0d8; border: 1px solid #d6e9c6; }
                .invalid { background-color: #f2dede; border: 1px solid #ebccd1; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .error { color: #a94442; }
                .warning { color: #8a6d3b; }
                .info { color: #31708f; }
            </style>
        </head>
        <body>
            <h1>Scene Validation Report</h1>
            <div class="summary {status}">
                <h2>Summary</h2>
                <p>Validation status: <strong>{status_text}</strong></p>
                <p>Validation time: {validation_time}</p>
                <p>Issues found: {issue_count} ({error_count} errors, {warning_count} warnings, {info_count} info)</p>
            </div>
            
            <h2>Issues</h2>
            <table>
                <tr>
                    <th>Severity</th>
                    <th>Timecode</th>
                    <th>Message</th>
                </tr>
                {issues_rows}
            </table>
        </body>
        </html>
        """
        
        # Generate issues table rows
        issues_rows = ""
        all_issues = [(e.severity, e.timecode, e.message) for e in self.errors + self.warnings + self.info]
        all_issues.sort(key=lambda x: x[1])  # Sort by timecode
        
        for severity, timecode, message in all_issues:
            issues_rows += f"""
                <tr class="{severity}">
                    <td>{severity.upper()}</td>
                    <td>{timecode}</td>
                    <td>{message}</td>
                </tr>
            """
        
        # Fill in template
        html = html.replace("{status}", "valid" if self.is_valid else "invalid")
        html = html.replace("{status_text}", "VALID" if self.is_valid else "INVALID")
        html = html.replace("{validation_time}", self.validation_time.strftime("%Y-%m-%d %H:%M:%S"))
        html = html.replace("{issue_count}", str(len(self.errors) + len(self.warnings) + len(self.info)))
        html = html.replace("{error_count}", str(len(self.errors)))
        html = html.replace("{warning_count}", str(len(self.warnings)))
        html = html.replace("{info_count}", str(len(self.info)))
        html = html.replace("{issues_rows}", issues_rows)
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _save_pdf_report(self, output_path: str) -> None:
        """Save the report in PDF format."""
        # For this implementation, we'll first create an HTML file and then convert it to PDF
        html_path = output_path.replace('.pdf', '_temp.html')
        self._save_html_report(html_path)
        
        try:
            # Using weasyprint to convert HTML to PDF
            # Note: This requires weasyprint to be installed
            from weasyprint import HTML
            HTML(html_path).write_pdf(output_path)
            os.remove(html_path)  # Clean up temporary HTML file
        except ImportError:
            logging.error("WeasyPrint is required for PDF generation. Install with 'pip install weasyprint'")
            raise


class SceneValidator:
    """
    A validator for checking scene structure and composition in video files.
    """
    
    def __init__(self, config_path: str = None, gemini_api_key: str = None, google_cloud_credentials: str = None):
        """
        Initialize the scene validator.
        
        Args:
            config_path: Path to validation configuration file
            gemini_api_key: API key for Gemini AI integration
            google_cloud_credentials: Path to Google Cloud credentials JSON file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
        # Set up Google Cloud Vision client if credentials are provided
        if google_cloud_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_cloud_credentials
            self.vision_client = vision.ImageAnnotatorClient()
        else:
            self.vision_client = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SceneValidator")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load validation configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading configuration from {config_path}: {str(e)}")
            return {}
    
    def validate_video(self, video_path: str, scene_markers: Optional[str] = None, 
                      output_format: str = "json") -> ValidationResult:
        """
        Validate a video file.
        
        Args:
            video_path: Path to the video file to validate
            scene_markers: Optional path to a JSON file with scene markers
            output_format: Format for the validation report ("json", "html", or "pdf")
            
        Returns:
            ValidationResult object containing the validation results
        """
        result = ValidationResult()
        
        # Check if video file exists
        if not os.path.exists(video_path):
            result.add_error("00:00:00.000", f"Video file not found: {video_path}")
            return result
        
        # Load scene markers if provided
        markers = None
        if scene_markers:
            try:
                with open(scene_markers, 'r') as f:
                    markers = json.load(f)
            except Exception as e:
                result.add_error("00:00:00.000", f"Error loading scene markers: {str(e)}")
        
        try:
            # Open the video file
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                result.add_error("00:00:00.000", "Failed to open video file")
                return result
            
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            result.add_info("00:00:00.000", f"Video loaded: {os.path.basename(video_path)}")
            result.add_info("00:00:00.000", f"Duration: {self._format_timecode(duration)}")
            result.add_info("00:00:00.000", f"Frame rate: {fps} fps")
            
            # Validate basic video properties
            self._validate_video_properties(video, result)
            
            # Validate scenes
            if markers:
                self._validate_scenes(video, markers, result)
            else:
                # If no markers provided, do automatic scene detection
                self._detect_and_validate_scenes(video, result)
            
            video.release()
            
        except Exception as e:
            result.add_error("00:00:00.000", f"Validation error: {str(e)}")
            self.logger.error(f"Exception during validation: {str(e)}", exc_info=True)
        
        return result
    
    def _format_timecode(self, seconds: float) -> str:
        """Format seconds as a timecode string (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    def _validate_video_properties(self, video: cv2.VideoCapture, result: ValidationResult) -> None:
        """Validate basic video properties."""
        # Check resolution
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        min_resolution = self.config.get("min_resolution", [640, 360])
        
        if width < min_resolution[0] or height < min_resolution[1]:
            result.add_warning("00:00:00.000", 
                              f"Low resolution: {width}x{height}. Minimum recommended: {min_resolution[0]}x{min_resolution[1]}")
        
        # Check frame rate
        fps = video.get(cv2.CAP_PROP_FPS)
        min_fps = self.config.get("min_fps", 24)
        
        if fps < min_fps:
            result.add_warning("00:00:00.000", 
                              f"Low frame rate: {fps} fps. Minimum recommended: {min_fps} fps")
    
    def _detect_and_validate_scenes(self, video: cv2.VideoCapture, result: ValidationResult) -> None:
        """Automatically detect scenes and validate them."""
        # Scene detection parameters
        threshold = self.config.get("scene_detection_threshold", 30)
        min_scene_length = self.config.get("min_scene_length", 1.0)  # seconds
        
        fps = video.get(cv2.CAP_PROP_FPS)
        min_frame_count = int(min_scene_length * fps)
        
        prev_frame = None
        scene_start_frame = 0
        current_frame_idx = 0
        scenes = []
        
        while True:
            ret, frame = video.read()
            if not ret:
                # End of video reached
                if current_frame_idx - scene_start_frame >= min_frame_count:
                    scenes.append((scene_start_frame, current_frame_idx))
                break
            
            # Convert frame to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate difference between frames
                diff = cv2.absdiff(gray, prev_frame)
                non_zero_count = np.count_nonzero(diff > threshold)
                
                # If significant difference detected, it might be a scene change
                if non_zero_count > (gray.shape[0] * gray.shape[1] * 0.05):  # 5% of pixels changed
                    if current_frame_idx - scene_start_frame >= min_frame_count:
                        scenes.append((scene_start_frame, current_frame_idx))
                    scene_start_frame = current_frame_idx
            
            prev_frame = gray.copy()
            current_frame_idx += 1
        
        # Validate each detected scene
        for i, (start_frame, end_frame) in enumerate(scenes):
            scene_start_time = start_frame / fps
            scene_end_time = end_frame / fps
            scene_duration = scene_end_time - scene_start_time
            
            # Report detected scene
            result.add_info(
                self._format_timecode(scene_start_time),
                f"Scene {i+1} detected: duration {scene_duration:.2f}s"
            )
            
            # Validate scene duration
            if scene_duration < min_scene_length:
                result.add_warning(
                    self._format_timecode(scene_start_time),
                    f"Scene {i+1} is too short: {scene_duration:.2f}s (min: {min_scene_length}s)"
                )
            
            # Analyze scene content by sampling frames
            self._analyze_scene_content(video, start_frame, end_frame, fps, result)
    
    def _validate_scenes(self, video: cv2.VideoCapture, markers: Dict[str, Any], result: ValidationResult) -> None:
        """Validate scenes based on provided scene markers."""
        scenes = markers.get("scenes", [])
        fps = video.get(cv2.CAP_PROP_FPS)
        
        for i, scene in enumerate(scenes):
            start_time = scene.get("start_time", 0)
            end_time = scene.get("end_time", 0)
            scene_type = scene.get("type", "unknown")
            
            # Convert times to frame indices
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Validate scene duration
            duration = end_time - start_time
            min_scene_length = self.config.get("min_scene_length", 1.0)
            
            if duration < min_scene_length:
                result.add_warning(
                    self._format_timecode(start_time),
                    f"Scene {i+1} ({scene_type}) is too short: {duration:.2f}s (min: {min_scene_length}s)"
                )
            
            # Check scene type-specific validations
            if scene_type == "dialogue" and duration < self.config.get("min_dialogue_scene_length", 3.0):
                result.add_warning(
                    self._format_timecode(start_time),
                    f"Dialogue scene {i+1} may be too short: {duration:.2f}s"
                )
            
            # Analyze scene content
            self._analyze_scene_content(video, start_frame, end_frame, fps, result)
    
    def _analyze_scene_content(self, video: cv2.VideoCapture, start_frame: int, end_frame: int, 
                              fps: float, result: ValidationResult) -> None:
        """Analyze the content of a scene by sampling frames."""
        # Sample frames for analysis (e.g., every second)
        sample_interval = int(fps)
        frames_to_analyze = list(range(start_frame, end_frame, sample_interval))
        
        # Ensure we analyze at least one frame per scene
        if not frames_to_analyze and start_frame < end_frame:
            frames_to_analyze = [start_frame]
        
        for frame_idx in frames_to_analyze:
            # Seek to the frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            
            if not ret:
                continue
            
            frame_time = frame_idx / fps
            timecode = self._format_timecode(frame_time)
            
            # Basic frame quality checks
            self._check_frame_quality(frame, timecode, result)
            
            # Use Google Cloud Vision API for advanced analysis if available
            if self.vision_client:
                self._analyze_frame_with_vision_api(frame, timecode, result)
    
    def _check_frame_quality(self, frame: np.ndarray, timecode: str, result: ValidationResult) -> None:
        """Check basic quality aspects of a frame."""
        # Check for extreme darkness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray)[0]
        
        if mean_brightness < self.config.get("min_brightness", 40):
            result.add_warning(timecode, f"Frame is too dark (brightness: {mean_brightness:.1f})")
        
        # Check for over-exposure
        if mean_brightness > self.config.get("max_brightness", 220):
            result.add_warning(timecode, f"Frame may be over-exposed (brightness: {mean_brightness:.1f})")
        
        # Check for blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.config.get("min_sharpness", 100):
            result.add_warning(timecode, f"Frame appears blurry (sharpness: {laplacian_var:.1f})")
    
    def _analyze_frame_with_vision_api(self, frame: np.ndarray, timecode: str, result: ValidationResult) -> None:
        """Use Google Cloud Vision API for advanced frame analysis."""
        try:
            # Convert frame to bytes for Vision API
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                return
            
            content = encoded_image.tobytes()
            image = vision.Image(content=content)
            
            # Detect faces
            response = self.vision_client.face_detection(image=image)
            faces = response.face_annotations
            
            # Check for composition issues with faces
            if faces:
                result.add_info(timecode, f"Detected {len(faces)} faces in frame")
                
                # Check for partially visible faces
                for face in faces:
                    if (face.detection_confidence < 0.8 or
                        face.bounding_poly.vertices[0].x < 0 or
                        face.bounding_poly.vertices[0].y < 0 or
                        face.bounding_poly.vertices[2].x > frame.shape[1] or
                        face.bounding_poly.vertices[2].y > frame.shape[0]):
                        result.add_warning(timecode, "Face may be partially outside frame")
            
            # Check for explicit content
            safe_search = self.vision_client.safe_search_detection(image=image).safe_search_annotation
            
            # Map likelihood string to numeric value for comparison
            likelihood_map = {
                vision.Likelihood.UNKNOWN: 0,
                vision.Likelihood.VERY_UNLIKELY: 1,
                vision.Likelihood.UNLIKELY: 2,
                vision.Likelihood.POSSIBLE: 3,
                vision.Likelihood.LIKELY: 4,
                vision.Likelihood.VERY_LIKELY: 5
            }
            
            # Check for potentially inappropriate content
            adult_score = likelihood_map.get(safe_search.adult, 0)
            violence_score = likelihood_map.get(safe_search.violence, 0)
            
            if adult_score >= 3:
                result.add_warning(timecode, f"Frame may contain adult content (score: {adult_score}/5)")
                
            if violence_score >= 3:
                result.add_warning(timecode, f"Frame may contain violent content (score: {violence_score}/5)")
            
        except Exception as e:
            self.logger.error(f"Error during Vision API analysis: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate scene structure and composition in video files")
    parser.add_argument("video_path", help="Path to the video file to validate")
    parser.add_argument("--scene-markers", help="Optional path to a JSON file with scene markers")
    parser.add_argument("--config", help="Path to validation configuration file")
    parser.add_argument("--output", default="validation_report.html", help="Output path for validation report")
    parser.add_argument("--format", choices=["json", "html", "pdf"], default="html", help="Output format")
    parser.add_argument("--gemini-key", help="Gemini API key (if not set in environment variable)")
    parser.add_argument("--gcp-credentials", help="Path to Google Cloud credentials JSON file")
    
    args = parser.parse_args()
    
    validator = SceneValidator(
        config_path=args.config,
        gemini_api_key=args.gemini_key,
        google_cloud_credentials=args.gcp_credentials
    )
    
    result = validator.validate_video(
        video_path=args.video_path,
        scene_markers=args.scene_markers,
        output_format=args.format
    )
    
    result.save_report(args.output)
    
    if result.is_valid:
        print(f"Validation PASSED. Report saved to {args.output}")
        exit(0)
    else:
        print(f"Validation FAILED with {len(result.errors)} errors. Report saved to {args.output}")
        for error in result.errors:
            print(f"  {error}")
        exit(1)