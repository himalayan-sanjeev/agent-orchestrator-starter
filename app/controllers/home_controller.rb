require 'faraday'
require 'json'

class HomeController < ApplicationController
  def index
    if params[:query].present?
      begin
        response = Faraday.get("http://localhost:8000/api/rag_query", { q: params[:query] })
        @response = JSON.parse(response.body)["answer"]
      rescue => e
        @response = "Error contacting AI backend: #{e.message}"
      end
    end
  end
end
