import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app, get_db, init_db
import os
import tempfile
import pandas as pd
import numpy as np
from werkzeug.security import generate_password_hash

class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set environment variable to use test database
        os.environ['APP_ENV'] = 'test'
    
    def setUp(self):
        # Create a temporary database
        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
        # Disable models for testing
        app.config['DISABLE_MODELS'] = True
        
        self.client = app.test_client()
        
        with app.app_context():
            init_db()
            
        # Clear any existing data
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM users")
            conn.commit()
            
            # Add test user
            cur.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                ('testuser', 'test@example.com', generate_password_hash('testpass'))
            )
            conn.commit()
            
        # Log in the test user
        response = self.client.post('/auth/login', json={
            'username_or_email': 'testuser',
            'password': 'testpass'
        })
        self.assertEqual(response.status_code, 200)

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(app.config['DATABASE'])

    def test_auth_flow(self):
        """Test the authentication flow"""
        # First logout to test unauthenticated flow
        self.client.post('/auth/logout')
        
        # Test that unauthenticated users are redirected to auth page
        response = self.client.get('/')
        self.assertEqual(response.status_code, 302)  # Expect redirect to auth page
        
        # Test logging in again
        response = self.client.post('/auth/login', json={
            'username_or_email': 'testuser',
            'password': 'testpass'
        })
        self.assertEqual(response.status_code, 200)
        
        # Test that auth endpoint returns user info when logged in
        response = self.client.get('/auth/me')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('user', data)
        
        # Test logout
        response = self.client.post('/auth/logout')
        self.assertEqual(response.status_code, 200)

    def test_user_registration(self):
        """Test user registration functionality"""
        response = self.client.post('/auth/signup', json={
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'newpass'
        })
        self.assertEqual(response.status_code, 200)
        
        # Check if user was added to database
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = ?", ('newuser',))
            user = cur.fetchone()
        self.assertIsNotNone(user)
        self.assertEqual(user['email'], 'new@example.com')

    def test_user_login(self):
        """Test user login functionality"""
        # Test with correct credentials
        response = self.client.post('/auth/login', json={
            'username_or_email': 'testuser',
            'password': 'testpass'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get('ok'))
        self.assertIn('user', data)
        
        # Test with incorrect credentials
        response = self.client.post('/auth/login', json={
            'username_or_email': 'testuser',
            'password': 'wrongpass'
        })
        self.assertEqual(response.status_code, 401)  # Unauthorized
        data = response.get_json()
        self.assertIn('error', data)

    def test_player_search(self):
        """Test player search functionality"""
        response = self.client.get('/api/players?search=Messi')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, dict)
        self.assertIn('items', data)
        self.assertIsInstance(data['items'], list)
        if len(data['items']) > 0:
            self.assertIn('name', data['items'][0])

    def test_compare_stats(self):
        """Test player stats comparison functionality"""
        response = self.client.get('/api/stats/top?position=ST')  # Compare top strikers
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, dict)
        self.assertIn('items', data)

    def test_semantic_search(self):
        """Test semantic search functionality"""
        response = self.client.get('/api/search/semantic?query=fast striker with good finishing&limit=5')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, dict)
        self.assertIn('items', data)
        self.assertIsInstance(data['items'], list)

if __name__ == '__main__':
    unittest.main()