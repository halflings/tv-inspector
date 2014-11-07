// error notification zone
var errorMessage = function(msg) {
    error_bar.text(msg);
    error_bar.slideDown(300).delay(3000).slideUp(300);
};

$(document).ready(function() {
    error_bar = $('#error-bar');

    $('#prediction-button').click(function() {
        var dialogText = $('#dialog-text').val();
        $.post('/predict', {dialog: dialogText}, function(data) {
            $('#prediction-result').text(data.prediction);
        });
    });
});