window.onload = function() {
    print(0);
}

function print(index, max) {
    if(index >= max) {
        return;
    }

    var elements = document.getElementsByTagName("svg"),
        len = elements.length;

    for(var j = 0; j < len; j++) {
        elements[j].style.display = (index == j) ? "block" : "none";
    }

    setTimeout(function() {
        print(index + 1, len);
    }, 300);
}