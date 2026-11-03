if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_csv> <output_csv>" >&2
    exit 1
fi

(head -n 1 "$1" && tail -n +2 "$1" | shuf -n 10000) > "$2"
