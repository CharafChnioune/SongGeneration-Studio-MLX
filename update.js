module.exports = {
  version: "4.0",
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "pip install -r requirements_mlx.txt"
      }
    }
  ]
}
