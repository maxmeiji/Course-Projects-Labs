<script
  src="https://code.jquery.com/jquery-3.4.1.js"
  integrity="sha256-WpOohJOqMqqyKL9FccASB9O0KwACQJpFTUBLTYOVvVU="
  crossorigin="anonymous"
></script>
<script 
  src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" 
  integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" 
  crossorigin="anonymous"
></script>
<script 
  src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" 
  integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" 
  crossorigin="anonymous"
></script>
<script src="https://cdn.jsdelivr.net/npm/sweetalert2@9"></script>
<?php
@session_start();

$time = $_SERVER['REQUEST_TIME'];
$timeout_duration = 1800;  // 設定30分鐘為timeout
if (!isset($_SESSION['is_login']) || ($time - $_SESSION['LAST_ACTIVITY']) > $timeout_duration) {
  header("Location:final.php");
  exit();
}
?>

