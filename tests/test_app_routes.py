import unittest
from io import BytesIO
from unittest.mock import patch

from app import app, normalize_stream_source
from video_detection import build_stream_candidates


class AppRoutesTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_welcome_page_loads(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome', response.data)

    def test_auth_page_loads(self):
        response = self.client.get('/auth')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sign In', response.data)

    def test_dashboard_page_loads(self):
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data)
        self.assertIn(b'Open live monitoring', response.data)

    def test_live_page_loads(self):
        response = self.client.get('/live')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Live Monitoring', response.data)
        self.assertIn(b'Start Webcam', response.data)

    def test_auth_signup_uses_firebase_handler(self):
        with patch('app._firebase_auth_request', return_value={
            'idToken': 'token',
            'email': 'builder@example.com',
            'displayName': 'Builder'
        }) as mock_request:
            response = self.client.post('/auth', data={
                'email': 'builder@example.com',
                'password': 'secret123',
                'mode': 'signup'
            })

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/dashboard')
        mock_request.assert_called_once()
        self.assertEqual(mock_request.call_args[0][0], 'signup')

    def test_predict_renders_uploaded_and_detected_image(self):
        with patch('app.process_image', return_value=('static/results/result.jpg', 3, 2, 1)):
            response = self.client.post(
                '/predict',
                data={'file': (BytesIO(b'image-bytes'), 'worker.jpg')},
                content_type='multipart/form-data'
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'/static/uploads/worker.jpg', response.data)
        self.assertIn(b'/static/results/result.jpg', response.data)
        self.assertIn(b'alt="Uploaded image"', response.data)
        self.assertIn(b'alt="Detection result"', response.data)

    def test_predict_renders_uploaded_and_detected_video(self):
        with patch('app.summarize_video', return_value=(4, 3, 1)):
            response = self.client.post(
                '/predict',
                data={'file': (BytesIO(b'video-bytes'), 'site.mp4')},
                content_type='multipart/form-data'
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'/static/uploads/site.mp4', response.data)
        self.assertIn(b'/processed_video_feed/site.mp4', response.data)
        self.assertIn(b'Site Status', response.data)
        self.assertNotIn(b'Workers Detected', response.data)
        self.assertNotIn(b'Safe Workers', response.data)
        self.assertNotIn(b'Compliance', response.data)
        self.assertNotIn(b'Detection Time', response.data)

    def test_cctv_monitoring_page_loads(self):
        response = self.client.get('/cctv')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CCTV Monitoring', response.data)

    def test_cctv_upload_uses_site_metadata(self):
        with patch('app.summarize_video', return_value=(4, 3, 1)):
            response = self.client.post(
                '/cctv',
                data={
                    'file': (BytesIO(b'video-bytes'), 'feed.mp4'),
                    'camera_id': 'CAM-01',
                    'site_zone': 'North Gate',
                    'incident_type': 'PPE review'
                },
                content_type='multipart/form-data'
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CAM-01', response.data)
        self.assertIn(b'North Gate', response.data)
        self.assertIn(b'PPE review', response.data)

    def test_cctv_stream_route_streams_response(self):
        with patch('app.generate_stream_frames', return_value=iter([b'frame-bytes'])):
            response = self.client.get('/cctv/stream?source=rtsp://camera.test/stream')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')

    def test_cctv_stream_route_normalizes_droidcam_source(self):
        with patch('app.generate_stream_frames', return_value=iter([b'frame-bytes'])) as mock_generate:
            response = self.client.get('/cctv/stream?source=localhost:4747')

        self.assertEqual(response.status_code, 200)
        mock_generate.assert_called_once_with('http://localhost:4747')

    def test_build_stream_candidates_includes_droidcam_paths(self):
        candidates = build_stream_candidates('http://localhost:4747')

        self.assertIn('http://localhost:4747', candidates)
        self.assertIn('http://localhost:4747/video', candidates)
        self.assertIn('http://localhost:4747/mjpegfeed', candidates)

    def test_normalize_stream_source_accepts_plain_droidcam_hosts(self):
        self.assertEqual(normalize_stream_source('localhost:4747/video'), 'http://localhost:4747/video')
        self.assertEqual(normalize_stream_source('192.168.1.25:4747/video'), 'http://192.168.1.25:4747/video')

    def test_webcam_feed_route_streams_response(self):
        with patch('app.generate_camera_frames', return_value=iter([b'frame-bytes'])):
            response = self.client.get('/webcam_feed?camera=1')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')

    def test_chat_api_returns_site_metrics_summary(self):
        with patch('app._assistant_history_stats', return_value={
            'records': [],
            'total_inspections': 5,
            'total_workers': 18,
            'total_safe': 14,
            'total_violations': 4,
            'safe_count': 3,
            'unsafe_count': 2,
            'avg_compliance': 88.5,
            'latest': {'file': 'zone-a.mp4', 'violations': 1, 'compliance': 92.0, 'date': '01-01-2026 10:00 AM', 'status': 'SAFE'},
        }):
            response = self.client.post('/api/chat', json={'message': 'Summarize my dashboard analytics'})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn('5', payload['reply'])
        self.assertIn('88.5%', payload['reply'])
        self.assertIn('zone-a.mp4', payload['reply'])

    def test_chat_api_handles_multi_topic_question(self):
        response = self.client.post('/api/chat', json={'message': 'Tell me about helmet and fall protection rules'})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn('Helmet And PPE Guidance', payload['reply'])
        self.assertIn('Fall Protection', payload['reply'])

    def test_chat_api_help_response_lists_capabilities(self):
        response = self.client.post('/api/chat', json={'message': 'What can you do?'})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn('How I Can Help', payload['reply'])
        self.assertIn('CCTV or webcam monitoring', payload['reply'])


if __name__ == '__main__':
    unittest.main()
