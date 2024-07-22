# Evaluation of 96-Well Data - Dockerized Streamlit Application

This project is a Dockerized Streamlit application designed for the evaluation of 96-well data. The application supports various types of analyses including conversion, comparison, and selectivity, provided that an index from A1 to H12 exists.

## Features

- **Conversion Analysis**: Transform and interpret data from 96-well plates.
- **Comparison Analysis**: Compare data across different wells or sets of wells.
- **Selectivity Analysis**: Assess selectivity metrics across the 96-well data set.

## Prerequisites

- Docker
- Docker Compose

## Getting Started

### Running the Application

To run the application using Docker Compose, navigate to the directory containing the `docker-compose.yml` file and run:

```sh
docker-compose up -d
