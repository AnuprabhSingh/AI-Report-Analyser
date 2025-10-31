"""
Flask REST API for Medical Report Interpretation System
Exposes endpoints for PDF upload and interpretation generation.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import sys
import tempfile
import json
import time
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor import MedicalReportExtractor
from src.predictor import ClinicalPredictor
from src.utils import format_clinical_output
from sklearn.metrics import confusion_matrix


# Initialize Flask app
# Set template folder to parent directory's templates folder
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize components
extractor = MedicalReportExtractor()
# Enable ML if models are available; prediction will gracefully fall back to rules otherwise
predictor = ClinicalPredictor(use_ml=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_class_labels(interpretations):
    """Extract class labels from interpretation text for charting."""
    import re
    
    labels = {}
    
    # LV_HYPERTROPHY from 'Interventricular Septum'
    ivs_text = interpretations.get('Interventricular Septum', '')
    if 'Normal' in ivs_text or 'normal' in ivs_text:
        labels['LV_HYPERTROPHY'] = 'Normal'
    elif 'Severe' in ivs_text:
        labels['LV_HYPERTROPHY'] = 'Severe'
    elif 'Moderate' in ivs_text:
        labels['LV_HYPERTROPHY'] = 'Moderate'
    elif 'Mild' in ivs_text:
        labels['LV_HYPERTROPHY'] = 'Mild'
    else:
        # If no clear label, assume Normal (common case)
        labels['LV_HYPERTROPHY'] = 'Normal'
    
    # LA_SIZE from 'Left Atrium'
    la_text = interpretations.get('Left Atrium', '')
    if 'enlarge' in la_text.lower():
        labels['LA_SIZE'] = 'Enlarged'
    elif 'Normal' in la_text or 'normal' in la_text:
        labels['LA_SIZE'] = 'Normal'
    else:
        # Default to Normal
        labels['LA_SIZE'] = 'Normal'
    
    # DIASTOLIC_FUNCTION from 'Diastolic Function'
    diastolic_text = interpretations.get('Diastolic Function', '')
    if 'Normal' in diastolic_text or 'normal' in diastolic_text:
        labels['DIASTOLIC_FUNCTION'] = 'Normal'
    elif 'Abnormal' in diastolic_text or 'abnormal' in diastolic_text:
        labels['DIASTOLIC_FUNCTION'] = 'Abnormal'
    else:
        # Default based on common patterns
        labels['DIASTOLIC_FUNCTION'] = 'Normal'
    
    return labels


@app.route('/')
def index():
    """Serve the frontend HTML page."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Medical Report Interpretation API',
        'version': '1.0.0'
    }), 200


@app.route('/api/interpret', methods=['POST'])
def interpret_report():
    """
    Main endpoint: Upload PDF and get clinical interpretation.
    
    Request:
        - file: PDF file (multipart/form-data)
        
    Response:
        - JSON with measurements and interpretations
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF allowed'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract data from PDF
        extracted_data = extractor.extract_from_pdf(filepath)
        
        if not extracted_data or not extracted_data.get('measurements'):
            return jsonify({
                'error': 'Failed to extract measurements from PDF',
                'details': 'No valid measurements found'
            }), 422
        
        # Generate interpretation
        measurements = extracted_data['measurements']
        patient_info = extracted_data['patient']
        
        interpretations = predictor.predict(measurements, patient_info)
        # Determine method and per-category sources
        method = 'ML-Based' if getattr(predictor, 'last_used_ml', False) else 'Rule-Based'
        try:
            sources = predictor.get_sources_for(interpretations)
        except Exception:
            sources = {k: ('ML' if getattr(predictor, 'last_used_ml', False) else 'Rule') for k in interpretations.keys()}
        
        # Clean up temporary file
        os.remove(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'file_name': filename,
            'patient': patient_info,
            'measurements': measurements,
            'interpretations': interpretations,
            'sources': sources,
            'method': method
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/interpret/json', methods=['POST'])
def interpret_from_json():
    """
    Interpret from already extracted JSON data.
    
    Request:
        - JSON body with measurements and patient info
        
    Response:
        - JSON with interpretations
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        measurements = data.get('measurements', {})
        patient_info = data.get('patient', {})
        
        if not measurements:
            return jsonify({'error': 'No measurements provided'}), 400
        
        # Generate interpretation
        interpretations = predictor.predict(measurements, patient_info)
        method = 'ML-Based' if getattr(predictor, 'last_used_ml', False) else 'Rule-Based'
        try:
            sources = predictor.get_sources_for(interpretations)
        except Exception:
            sources = {k: ('ML' if getattr(predictor, 'last_used_ml', False) else 'Rule') for k in interpretations.keys()}
        
        response = {
            'success': True,
            'measurements': measurements,
            'patient': patient_info,
            'interpretations': interpretations,
            'sources': sources,
            'method': method
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/parameters', methods=['GET'])
def get_supported_parameters():
    """
    Get list of supported measurement parameters.
    
    Response:
        - JSON with parameter list and normal ranges
    """
    parameters = {
        'EF': {
            'name': 'Ejection Fraction',
            'unit': '%',
            'normal_range': '55-70'
        },
        'LVID_D': {
            'name': 'LV Internal Diameter (Diastole)',
            'unit': 'cm',
            'normal_range': '3.9-5.9'
        },
        'LVID_S': {
            'name': 'LV Internal Diameter (Systole)',
            'unit': 'cm',
            'normal_range': '2.2-4.0'
        },
        'IVS_D': {
            'name': 'Interventricular Septum (Diastole)',
            'unit': 'cm',
            'normal_range': '0.6-1.0'
        },
        'LVPW_D': {
            'name': 'LV Posterior Wall (Diastole)',
            'unit': 'cm',
            'normal_range': '0.6-1.0'
        },
        'LA_DIMENSION': {
            'name': 'Left Atrium Dimension',
            'unit': 'cm',
            'normal_range': '2.7-4.0'
        },
        'MV_E_A': {
            'name': 'Mitral Valve E/A Ratio',
            'unit': 'ratio',
            'normal_range': '0.8-1.5'
        },
        'FS': {
            'name': 'Fractional Shortening',
            'unit': '%',
            'normal_range': '25-45'
        },
        'LV_MASS': {
            'name': 'LV Mass',
            'unit': 'gm',
            'normal_range': '67-224'
        }
    }
    
    return jsonify({
        'parameters': parameters,
        'total_count': len(parameters)
    }), 200


@app.route('/api/batch', methods=['POST'])
def batch_process():
    """
    Batch process multiple PDF files.
    
    Request:
        - files: Multiple PDF files
        
    Response:
        - JSON with results for each file
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Extract and interpret
                extracted_data = extractor.extract_from_pdf(filepath)
                
                if extracted_data and extracted_data.get('measurements'):
                    interpretations = predictor.predict(
                        extracted_data['measurements'],
                        extracted_data['patient']
                    )
                    
                    # Extract class labels for charting
                    class_labels = extract_class_labels(interpretations)
                    
                    method = 'ML-Based' if getattr(predictor, 'last_used_ml', False) else 'Rule-Based'
                    try:
                        sources = predictor.get_sources_for(interpretations)
                    except Exception:
                        sources = {k: ('ML' if getattr(predictor, 'last_used_ml', False) else 'Rule') for k in interpretations.keys()}

                    results.append({
                        'file_name': filename,
                        'patient': extracted_data['patient'],
                        'measurements': extracted_data['measurements'],
                        'interpretations': interpretations,
                        'sources': sources,
                        'method': method,
                        'class_labels': class_labels,
                        'status': 'success'
                    })
                else:
                    errors.append({
                        'file_name': filename,
                        'error': 'No measurements extracted'
                    })
                
                # Clean up
                os.remove(filepath)
            
            except Exception as e:
                errors.append({
                    'file_name': file.filename,
                    'error': str(e)
                })
        else:
            errors.append({
                'file_name': file.filename,
                'error': 'Invalid file type'
            })
    
    return jsonify({
        'success': len(results),
        'failed': len(errors),
        'results': results,
        'errors': errors
    }), 200


def create_app():
    """Factory function to create Flask app."""
    return app


@app.route('/api/model-metrics', methods=['GET'])
def model_metrics():
    """
    Return cached or freshly computed model comparison metrics including
    accuracy, precision, recall, F1, CV scores, timings and confusion matrices
    for each supported category and algorithm.

    Query Params:
        - force=true to recompute instead of using cached file

    Response JSON shape:
        {
          "generated_at": <timestamp>,
          "categories": {
             <CATEGORY>: {
                "best_algorithm": <str>,
                "algorithms": {
                   <ALGO_NAME>: {
                      "accuracy": <float>,
                      "precision": <float>,
                      "recall": <float>,
                      "f1_score": <float>,
                      "cv_mean": <float>,
                      "cv_std": <float>,
                      "train_time": <float>,
                      "predict_time": <float>,
                      "confusion_matrix": {"labels": [..], "matrix": [[..]]}
                   }
                }
             }
          }
        }
    """
    try:
        force = str(request.args.get('force', 'false')).lower() == 'true'

        # Paths
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metrics_path = os.path.join(root_dir, 'data', 'processed', 'algorithm_metrics.json')
        dataset_path = os.path.join(root_dir, 'data', 'processed', 'training_dataset.json')

        # Serve cached results if available
        if not force and os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                cached = json.load(f)
            return jsonify(cached), 200

        if not os.path.exists(dataset_path):
            return jsonify({
                'error': 'Dataset not found',
                'details': f'Missing training dataset at {dataset_path}'
            }), 404

        # Lazy import to avoid overhead if endpoint unused
        try:
            from compare_algorithms import AlgorithmComparator
        except Exception as e:
            return jsonify({'error': 'Failed to import comparator', 'details': str(e)}), 500

        comparator = AlgorithmComparator()
        categories = ['LV_HYPERTROPHY', 'LA_SIZE', 'DIASTOLIC_FUNCTION']
        results = comparator.compare_algorithms(dataset_path, categories)

        output = {
            'generated_at': time.time(),
            'categories': {},
            'global_stats': {
                'avg_train_time': 0.0,
                'avg_predict_time': 0.0,
                'total_samples': 0
            }
        }        # Build metrics with confusion matrices and additional stats
        for category, algo_results in (results or {}).items():
            cat_payload = {
                'algorithms': {}, 
                'best_algorithm': None,
                'aggregated_stats': {
                    'avg_precision': 0.0,
                    'avg_recall': 0.0,
                    'std_precision': 0.0,
                    'std_recall': 0.0,
                    'total_samples': 0
                }
            }
            best_acc = -1.0

            if not algo_results:
                output['categories'][category] = cat_payload
                continue

            for algo_name, res in algo_results.items():
                if not res:
                    continue

                # Convert to lists (ensure JSON serializable)
                y_true = [str(x) for x in list(res['actual'])]
                y_pred = [str(x) for x in list(res['predictions'])]
                labels = sorted(list(set(y_true) | set(y_pred)))

                try:
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    cm_list = cm.tolist()
                except Exception:
                    cm_list = []

                # Extract base metrics
                algo_metrics = {
                    'accuracy': float(res.get('accuracy', 0.0)),
                    'precision': float(res.get('precision', 0.0)),
                    'recall': float(res.get('recall', 0.0)),
                    'f1_score': float(res.get('f1_score', 0.0)),
                    'cv_mean': float(res.get('cv_mean', 0.0)),
                    'cv_std': float(res.get('cv_std', 0.0)),
                    'train_time': float(res.get('train_time', 0.0)),
                    'predict_time': float(res.get('predict_time', 0.0)),
                    'confusion_matrix': {
                        'labels': labels,
                        'matrix': cm_list
                    }
                }

                # Update aggregated stats for the category
                n_samples = len(y_true)
                cat_payload['aggregated_stats']['total_samples'] += n_samples
                cat_payload['aggregated_stats']['avg_precision'] += algo_metrics['precision'] * n_samples
                cat_payload['aggregated_stats']['avg_recall'] += algo_metrics['recall'] * n_samples
                
                # Update global timing stats
                output['global_stats']['avg_train_time'] += algo_metrics['train_time']
                output['global_stats']['avg_predict_time'] += algo_metrics['predict_time']
                output['global_stats']['total_samples'] += n_samples

                cat_payload['algorithms'][algo_name] = algo_metrics

                if algo_metrics['accuracy'] > best_acc:
                    best_acc = algo_metrics['accuracy']
                    cat_payload['best_algorithm'] = algo_name

            # Normalize aggregated stats for the category if we have samples
            if cat_payload['aggregated_stats']['total_samples'] > 0:
                n_total = cat_payload['aggregated_stats']['total_samples']
                cat_payload['aggregated_stats']['avg_precision'] /= n_total
                cat_payload['aggregated_stats']['avg_recall'] /= n_total

                # Calculate standard deviations
                precision_var = sum(
                    (algo_res['precision'] - cat_payload['aggregated_stats']['avg_precision']) ** 2
                    for algo_name, algo_res in cat_payload['algorithms'].items()
                ) / len(cat_payload['algorithms'])
                recall_var = sum(
                    (algo_res['recall'] - cat_payload['aggregated_stats']['avg_recall']) ** 2
                    for algo_name, algo_res in cat_payload['algorithms'].items()
                ) / len(cat_payload['algorithms'])

                cat_payload['aggregated_stats']['std_precision'] = precision_var ** 0.5
                cat_payload['aggregated_stats']['std_recall'] = recall_var ** 0.5

            output['categories'][category] = cat_payload

        # Normalize global timing stats
        total_algos = sum(len(cat['algorithms']) for cat in output['categories'].values())
        if total_algos > 0:
            output['global_stats']['avg_train_time'] /= total_algos
            output['global_stats']['avg_predict_time'] /= total_algos

        # Cache the results
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(output, f, indent=2)

        return jsonify(output), 200

    except Exception as e:
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Medical Report Interpretation API Server")
    print("=" * 60)
    print("Starting server...")
    print("API Documentation:")
    print("  POST /api/interpret - Upload PDF and get interpretation")
    print("  POST /api/interpret/json - Send JSON data for interpretation")
    print("  GET  /api/parameters - Get supported parameters")
    print("  POST /api/batch - Batch process multiple PDFs")
    print("  GET  /api/model-metrics - Get model performance metrics and confusion matrices")
    print("  GET  /health - Health check")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
