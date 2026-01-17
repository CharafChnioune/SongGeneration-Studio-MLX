module.exports = {
  version: "4.0",
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python -m pip install --upgrade pip",
          "pip install -r requirements_mlx.txt"
        ]
      }
    }
  ]
}
