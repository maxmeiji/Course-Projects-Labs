<?php

require_once 'db.php';
@session_start();


function check_has_email($card)
{
	
	
  $result = null;

  
  $sql = "SELECT * FROM `user` WHERE `card` = '{$card}';";

  
  $query = mysqli_query($_SESSION['link'], $sql);

  
  if ($query)
  {
    
    if (mysqli_num_rows($query) >= 1)
    {

      $result = true;
    }

    //釋放資料庫查詢到的記憶體
    mysqli_free_result($query);
  }
  else
  {
    echo "{$sql} 語法執行失敗，錯誤訊息：" . mysqli_error($_SESSION['link']);
  }

  //回傳結果
  return $result;
}


$check = check_has_email($_POST['cd']);

if($check)
{
	//若為true 代表有使用者以重複
	echo 'no';
}
else
{
	//若為 null 或者 false 代表沒有使用者，可以註冊
	echo 'yes';	
}

?>