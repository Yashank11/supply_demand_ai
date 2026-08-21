def get_custom_css() -> str:
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main background and containers */
        .stApp {
            background: linear-gradient(135deg, #0B0F19 0%, #111827 50%, #0B0F19 100%);
            color: #F3F4F6;
        }
        
        /* Premium Card styling */
        .scout-card {
            background: rgba(17, 24, 39, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s ease;
        }
        
        .scout-card:hover {
            border-color: rgba(99, 102, 241, 0.35);
            box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
        }
        
        /* Glowing Metric Card */
        .metric-card {
            background: linear-gradient(145deg, rgba(31, 41, 55, 0.6) 0%, rgba(17, 24, 39, 0.8) 100%);
            border-radius: 12px;
            padding: 16px 20px;
            border-left: 4px solid #6366F1;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            margin-bottom: 12px;
        }
        
        .metric-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9CA3AF;
            margin-bottom: 4px;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #F9FAFB;
            line-height: 1.2;
        }
        
        .metric-subtitle {
            font-size: 0.8rem;
            color: #6EE7B7;
            margin-top: 4px;
            font-weight: 500;
        }
        
        /* Badges */
        .badge-critical {
            background-color: rgba(239, 68, 68, 0.2);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
        }
        
        .badge-warning {
            background-color: rgba(245, 158, 11, 0.2);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.4);
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
        }
        
        .badge-optimal {
            background-color: rgba(16, 185, 129, 0.2);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
        }
        
        /* Header Hero */
        .hero-banner {
            background: linear-gradient(90deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3);
        }
        
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FFFFFF 0%, #A5B4FC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        
        .hero-sub {
            font-size: 0.95rem;
            color: #CBD5E1;
            margin-top: 6px;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0D1117;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
            background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%);
            border-color: #818CF8;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
            transform: translateY(-1px);
        }
    </style>
    """
