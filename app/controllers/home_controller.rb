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

  def crew_summary
    if params[:topic].present?
      begin
        response = Faraday.get("http://localhost:8000/api/crew_summary", { topic: params[:topic] })
        @crew_summary = JSON.parse(response.body)
      rescue => e
        @crew_summary = { error: e.message }
      end
    end
    render :index
  end
end
