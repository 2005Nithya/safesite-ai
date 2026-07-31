import unittest
from io import BytesIO
from unittest.mock import patch

from app import app


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

    def test_webcam_feed_route_streams_response(self):
        with patch('app.generate_camera_frames', return_value=iter([b'frame-bytes'])):
            response = self.client.get('/webcam_feed?camera=1')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')


if __name__ == '__main__':
    unittest.main()
