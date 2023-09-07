if ! pip freeze | grep 'box2d-py'
then
    # build box2d_py from source
    cd "$FORNIX_FOLDER/repos/box2d_py"
    python setup.py \
        build \
        install
fi