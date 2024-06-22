class CSSGenerator:
    color = "#2A62D2"

    @classmethod
    def generate_custom_css(cls, max_tokens):
        custom_css = f"""
            <style>
                /* Tick bar background color */
                div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {{
                    background: rgb(1 1 1 / 0%);
                }}

                /* Slider knob background color */
                div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {{
                    background-color: {cls.color};
                }}

                /* Slider value text color */
                div.stSlider > div[data-baseweb="slider"] > div > div > div > div {{
                    color: {cls.color};
                }}         
                /* Button hover, active, and focus styles */
                div.stButton > button:hover,
                div.stButton > button:active,
                div.stButton > button:focus {{
                    background-color: {cls.color} !important;
                    border-color: {cls.color} !important;
                    color: white !important;
                }}      
        
                /* Slider gradient background based on max_tokens */
                div.stSlider > div[data-baseweb="slider"] > div > div {{
                    background: linear-gradient(to right, 
                        rgb(42, 98, 210) 0%, 
                        rgb(42, 98, 210) {max_tokens}%, 
                        rgba(214, 218, 221, 0.25) {max_tokens}%, 
                        rgba(214, 218, 221, 0.25) 100%);
                }}
            </style>
            """
        return custom_css
