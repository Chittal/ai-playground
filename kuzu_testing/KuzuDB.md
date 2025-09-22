Command to open Kuzu DB explorer
```
docker run --rm -p 8000:8000 -v "${PWD}:/database" -e KUZU_FILE="/example.kuzu" kuzudb/explorer:latest
```