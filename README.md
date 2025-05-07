# Vendor Search & Recommendation

A Streamlit application that helps users find and get recommendations for vendors based on products/services and location.

## Features

- Search for vendors by product/service and location
- Get AI-powered recommendations for top vendors
- View and download search results
- User-friendly wizard interface

## Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - `GOOGLE_CSE_API_KEY`: Google Custom Search API key
   - `GOOGLE_CSE_ID`: Google Custom Search Engine ID
   - `GEMINI_API_KEY`: Google Gemini API key

4. Run the application:
   ```
   streamlit run app.py
   ```

## Deployment on Render

This application can be deployed to Render using the included `render.yaml` configuration:

1. Fork this repository to your GitHub account
2. Connect your GitHub repository to Render
3. Create a new Blueprint on Render using this repository
4. Set the required environment variables in the Render dashboard:
   - `GOOGLE_CSE_API_KEY`
   - `GOOGLE_CSE_ID`
   - `GEMINI_API_KEY`

## Adding a Logo

Place a file named `logo.png` in the same directory as `app.py` to display your custom logo.

## License

MIT 