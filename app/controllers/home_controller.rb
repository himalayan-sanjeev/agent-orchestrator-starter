class HomeController < ApplicationController
  def index
    response = Faraday.get('http://localhost:8000/api/ai_response')
    @ai_message = JSON.parse(response.body)['message'] if response.success?
  rescue StandardError => e
    @ai_message = "Error: #{e.message}"
  end
end
